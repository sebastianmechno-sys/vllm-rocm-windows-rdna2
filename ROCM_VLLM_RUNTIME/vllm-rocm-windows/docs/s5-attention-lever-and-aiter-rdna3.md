# S5 native paged attention, the real decode lever, and the AITER-on-RDNA3 intent

Status as of 2026-07-02. This is an honest write-up including the negative results, because they
define what to build next.

## TL;DR

- We built a **native HIP `paged_attention_v1` + a sliding-window mask** for gfx1100 (from vLLM's
  generic wave32 `csrc/attention/`, NOT the gfx9 wave64/MFMA `csrc/rocm/attention.cu`). In isolation it
  is **~3.2x faster** than the Triton decode kernel it would replace (`kernel_paged_attention_2d`) and
  **numerically correct** (rel err ~5e-4, sliding-window matches vLLM's `(context_len-seq_offset) <
  SLIDING_WINDOW`).
- **But end-to-end it REGRESSES** when integrated via the ROCM_ATTN backend: **-9% on gemma-4-26B, -5%
  on ERNIE-4.5-21B** (the latter on a clean, spill-free, uniform-head-128, coherent run). The isolated
  3.2x does not translate because the ROCM_ATTN decode PATH around the kernel (v0 `reshape_and_cache`
  write + the `chunked_prefill_paged_decode` wrapper + ROCM_ATTN metadata) is heavier than TRITON_ATTN's
  fused `do_rope_and_kv_cache_update` + `unified_attention`.
- **What the profiler said was wrong.** `torch.profiler` self-device-time on this TheRock Windows
  torch-rocm build mis-attributes time to zero-kernel view ops (`as_strided`/`slice`/`view`) even under
  cudagraph. The eager-derived claim "MoE dispatch plumbing is ~50% of decode" is an artifact.
- **The real decode lever, by ABLATION (no-op a component, measure the tok/s delta under cudagraph):**
  **attention COMPUTE ~27%** (the single biggest), dense bf16 MLP GEMV ~3.3%, MoE expert compute ~0%.
  So attention is where the time is -- but only a kernel swap that keeps the light TRITON_ATTN path can
  capture it.

## What was built (this session)

- `patches/vllm/native-attn-sliding-window.patch` -- threads a `sliding_window` param through
  `csrc/attention/attention_kernels.cuh` (the QK mask: `sw_mask = (sw>0) && ((seq_len-1-token) >= sw)`
  -> logit -FLT_MAX -> excluded from softmax + V accum), `paged_attention_v1.cu` (device func -> global
  kernel -> LAUNCH macro -> launcher -> public op), and `ops.h`. v2 left intact. `sliding_window<=0`
  keeps the original full-attention behavior; perf unchanged (mask is a couple of ALU ops).
- `experiments/vllm_c_ext/build_attn_c.py` + `win_attn_bindings.cu` + `build_attn_run.bat` -> builds
  `vllm_win_attn_C.pyd` (torch.ops._C.paged_attention_v1/v2, `TORCH_LIBRARY_FRAGMENT(_C)` so it
  coexists with `vllm_win_C`). Head sizes up to 256 (incl. 128, 256); block_size 8/16/32.
- `experiments/vllm_c_ext/build_cache_c.py` + `win_cache_bindings.cu` -> `vllm_win_cache_C.pyd` now
  exposes BOTH `reshape_and_cache` (v0 paged layout, for the ROCM_ATTN write path) and
  `reshape_and_cache_flash`.
- `windows_rocm_plugin/vllm_windows_rocm/cops.py` -- `_install_attn_C()` loads the attn pyd (opt-in
  `VLLM_WIN_ATTN_NATIVE=1`); `maybe_patch_s5_decode()` (call AFTER `from vllm import LLM`) monkeypatches
  `chunked_prefill_paged_decode`: for pure decode (`max_query_len==1`) with `head_size in (128,256)`,
  `kv_cache_dtype=="auto"`, it calls the native `paged_attention_v1` instead of the Triton kernel; it
  also disables `use_rocm_custom_paged_attention` (on gfx1x head-128 that gate PASSES and calls
  `ops.paged_attention_rocm == torch.ops._rocm_C.paged_attention`, the gfx9 MFMA kernel we do NOT have).
- `windows_rocm_plugin/sitecustomize.py` -- **must be copied to `site-packages/`.** Applies the
  single-process torch.distributed shim at EVERY interpreter start, so vLLM's model-inspection
  SUBPROCESS (`python -m vllm.model_executor.models.registry`, which does not load the plugin) works on
  un-cached architectures (e.g. ERNIE-4.5). Without it: `ModuleNotFoundError: torch._C._distributed_c10d`.
  Caveat: `torchdist_shim.apply()` also calls `cops.install()`, so this makes every python load the
  native kernels AND evaluates the opt-in `VLLM_WIN_*` env flags AS OF STARTUP -> set them in the OS env
  before launching python, not only in-script.

## Clean measurements (ERNIE-4.5-21B-A3B, spill-free)

gemma-4-26B-A4B-AWQ is 17GB on disk; on the 20GB RX 7900 XT that leaves ~0.7 GiB free -> WDDM spills to
shared DRAM -> gemma tok/s were borderline/contaminated. **ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit**
(compressed-tensors W4A16 gs32 MoE, head_dim 128 uniform, no sliding, ~14GB -> 3+ GiB free -> no spill)
is the clean bench. `run/s5_bench/perf_ernie.py`, cudagraph, best-of-3:

| config | tok/s |
| --- | --- |
| vanilla TRITON_ATTN | 62.7 |
| optimized (MoE-decode GEMV + wvSplitK) TRITON_ATTN | **79.2** |
| optimized + S5 (ROCM_ATTN + native paged_attention_v1, coherent, native fired 1960x) | 75.2 (**-5%**) |

Ablation (`run/s5_bench/perf_ablation.py`, no-op a component, measure the tok/s ceiling under cudagraph;
this is the RELIABLE method since the profiler is not): attention COMPUTE ~+27%, dense bf16 GEMV +3.3%,
MoE experts ~0%.

## The only remaining path to the ~27% attention lever

**A flash-layout native kernel.** Keep TRITON_ATTN's light fused path (rope+KV-write, its metadata) and
swap ONLY `unified_attention` for a native kernel that reads TRITON_ATTN's FLASH KV cache
`[num_blocks, block_size, num_kv_heads, head_size]` directly (rewrite `paged_attention_v1`'s K/V loads
from the v0 `x`-split layout to flash order). This avoids the ROCM_ATTN path overhead that killed the
ROCM_ATTN integration. Invasive kernel rewrite + coalescing/perf risk; unproven. This is the concrete
next experiment if we resume the attention lever.

## AITER on RDNA3 -- the intent

The standing goal: **if a kernel AMD ships only for CDNA/RDNA4 (AITER) does not exist for RDNA3, we build
it.** AITER's headline decode wins are `fmha_v3` (paged attention) and MLA -- ASM-tuned `.co` VLIW blobs
under `hsa/gfx942|950|1250` with no gfx1100 blob, MFMA-locked, and `setup.py` forces
`AITER_TRITON_ONLY=True`/`ENABLE_CK=False` on win32. So a "1:1 port" is not portable for the parts that
matter.

**Our S5 native paged attention IS the RDNA3-native equivalent of AITER's `fmha_v3` decode kernel** --
generic wave32, WMMA-friendly, 100% gfx1100. This session proved the KERNEL is a real 3.2x, so the AITER
premise ("these kernels are buildable on RDNA3") holds. What this session ALSO proved is that the WIN is
in the *kernel*, but capturing it e2e needs the *right host path* (flash-layout swap, not the heavy
ROCM_ATTN backend). The AITER-on-RDNA3 program therefore continues as: build the RDNA3-native decode
attention as a flash-layout drop-in for `unified_attention`, then (later) a wave32 WMMA GEMM for the
prefill/batch path. Everything stays opt-in and default-off; nothing here changes the working
TRITON_ATTN path.

## Flash-kernel result (2026-07-02): done, and it does NOT win e2e -- the native-attention advantage is head-256-specific

We built the flash-layout kernel described above: `experiments/vllm_c_ext/paged_attention_flash.cu` ->
`torch.ops._C.paged_attention_flash`, a fresh decode-only kernel that reads TRITON_ATTN's flash KV layout
`[num_blocks, block_size, num_kv_heads, head_size]` DIRECTLY (thread-group Phase-1 QK + unrolled/vectorized
loads, in-kernel sliding mask). `cops.maybe_patch_flash_decode()` (`VLLM_WIN_FLASH_ATTN=1`) monkeypatches
`unified_attention` so a pure-decode call (`max_seqlen_q==1`, head 128/256) runs it, keeping TRITON_ATTN's
light fused path -- NO ROCM_ATTN overhead. It is correct (rel ~3e-4 fp16, sliding OK) and fires (1960x on
an ERNIE run) with coherent output.

**But e2e on ERNIE-4.5-21B (head 128, clean VRAM): 58.8 tok/s vs 79.2 baseline = -26%.** The kernel is
~82us (head 128, seq 512) in isolation; the Triton `unified_attention` is ~40us at head 128 -- i.e. Triton
is already well-tuned there and our hand kernel is ~2x slower. The earlier "native is 3.2x faster than
unified" (that motivated this whole line) was measured at head 256 (gemma), where `unified_attention` is
pathologically slow (284us). **So the native-decode-attention advantage is HEAD-256-SPECIFIC, not general.**
At the common head-128 (ERNIE / llama / qwen) the Triton unified is good; even a v0-perfect flash kernel
(~40us) would only TIE it (no e2e win). At head 256 the flash kernel would win ~1.7x -- but the only head-256
MoE we have (gemma-4-26B) overfills VRAM and spills, so it cannot be measured cleanly.

Net: the flash path is architecturally correct and the kernel is banked (reusable for a head-256 model on
adequate VRAM/hardware), but it yields no clean decode win on the models we can measure. The AITER-on-RDNA3
program's decode-attention piece is therefore parked: `unified_attention` is not the beatable target at
head 128 that it is at head 256.
