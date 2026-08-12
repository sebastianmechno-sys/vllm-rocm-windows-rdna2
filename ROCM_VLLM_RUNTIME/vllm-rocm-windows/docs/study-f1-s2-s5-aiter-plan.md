The nested vLLM AGENTS.md is upstream-contribution policy for `vllm-project/vllm` — this is a read-only study for the user's own Windows-ROCm fork, no PRs to upstream, so it does not constrain this task.

I have everything verified. The key correction to the studies: on this Windows torch-rocm build, native ops register under the **`CUDA` dispatch key** (confirmed in cops.py:183,263,282), not `torch::kHIP` — the S2 study's `torch::kHIP` recommendation would silently fail to bind. Now synthesizing the final plan.

---

# vLLM Windows-ROCm gfx1100 — Native-Kernel Build/Study Plan (F1 → S2 → S5 + AITER verdict)

## 0. One correction up front (load-bearing, verified against your tree)

The S2 and S5 studies both recommend registering native ops for `torch::kHIP`. **On your stack that is wrong and would silently no-op.** Your own loader `cops.py` registers every fallback under the **`CUDA` dispatch key** (lines 183, 263, 282), because torch-rocm on Windows presents HIP devices as `torch.cuda` / `torch::kCUDA`. The existing native `_C`/`_moe_C`/`_rocm_C` pyds you already shipped bind under `TORCH_LIBRARY(_C)` + `torch::kCUDA` and work. So: **keep `torch::kCUDA` everywhere. Do not add `torch::kHIP`.** This is the single most important fix to the two sub-studies.

Also confirmed: `reshape_and_cache_flash` is registered CUDA-only at `torch_bindings.cpp:723-724`; the gate `use_rocm_custom_paged_attention` at `rocm.py:247` requires `head_size==128` (line 276) and `sliding_window in (0,(-1,-1))` (line 274), so gemma-4 (head-256 + sliding-1024) is excluded exactly as described.

---

## 1. F1 → S2 → S5 CONCRETE BUILD PLAN

Dependency order is **hard**: F1 is a prerequisite for S5 (block-reduce in the softmax finalize) and is a "free" side-upgrade for S3/Q1/Q2/Q6. S2 is independent of F1 but should go second because it is the lowest-risk way to prove the "register a generic csrc op under kCUDA and route the v1 backend to it" wiring that S5 then reuses at much larger scale.

### F1 — wave32 block-reduce / block-scan header (do FIRST)

**Create:**
- `vllm/csrc/rocm_wave32_reduce.cuh` (~100–150 LOC). Template `block_reduce<T,Op>` + `block_scan` (ExclusiveSum/InclusiveSum). Copy the proven pattern from `csrc/attention/attention_kernels.cuh` `block_sum` (warp `__shfl_xor` tree → LDS[32] → single-warp finalize). `WARP_SIZE` comes from `cuda_compat.h` (already `32` on gfx11).

**Adapt:**
- `vllm/csrc/cub_helpers.h` — add a `#ifdef USE_ROCM` branch that `#include`s the new header and typedefs `CubAddOp`/`CubMaxOp`/`CubArgMax` to the wave32 ops. **Kernel .cu files need zero edits** — they already go through `cub_helpers.h`.

**Wiring/loader:** none. F1 is header-only; it takes effect when any consuming kernel (.pyd) is rebuilt.

**Validation:** standalone microkernel comparing to a CPU reference for SUM(float), MAX(float), ARGMAX(KeyValuePair), ExclusiveSum(int32) across block sizes 128/256/512/1024. Integer ops must be bit-exact; float within <1 ULP. This is the only place you can cheaply prove correctness in isolation — do it before S5 depends on it.

**Effort:** 0.5–1 day. **Risk: LOW** (the exact pattern already runs on your gfx1100 in paged-attn).

**Honesty note:** F1 alone buys **~0 tok/s** today — nothing in your current 56.8 tok/s decode path calls CUB (your MoE align/sum/topk are the torch/native fallbacks in cops.py + `_moe_C`). F1's value is strictly as an **unblocker** for S5 and the S3/Q-tier norm+quant kernels. Sequence it first, but don't expect it to move the number.

### S2 — native `reshape_and_cache_flash` (do SECOND, proving ground for the wiring)

**Adapt:**
- `vllm/csrc/cache_kernels.cu` — hipify-clean. The only CUDA-ism on a non-decode path is `cudaMemcpy`→`hipMemcpy` in `swap_blocks` (not on the single-GPU decode path). The `flash` kernel's `lane = threadIdx.x & 31` / `>>5` is already wave32-correct. FP8 path uses `fp8::scaled_convert` from `quantization/w8a8/fp8/amd/quant_utils.cuh` (already `USE_ROCM`-guarded).
- `vllm/csrc/torch_bindings.cpp:723` — **keep `torch::kCUDA`** (do NOT switch to kHIP). The reason the op is "missing" on your build isn't the dispatch key; it's that you have not compiled a `_C_cache_ops` pyd at all yet.

**Create:**
- A `win_cache_bindings.cu` (mirror your existing `win_*_bindings.cu`) that does `TORCH_LIBRARY(_C_cache_ops, ...)` + `.impl(..., torch::kCUDA, ...)` for `reshape_and_cache_flash` (and optionally `reshape_and_cache`), so the op lands in the `_C_cache_ops` namespace vLLM calls.
- A build entry in `experiments/vllm_c_ext/build_c_ext.py` producing `vllm_win_cache_C.pyd`.

**Wiring/integration point:**
1. **Loader:** add a `_load_native`-style block in `cops.py` (clone the `_load_native()` at line 30) to `torch.ops.load_library` the new `vllm_win_cache_C*.pyd` from a `VLLM_WIN_CACHE_DIR`. This is the exact pattern already used for `_moe_C` at lines 159–167.
2. **Call site:** `vllm/v1/attention/backends/triton_attn.py` `do_kv_cache_update` (~line 597) currently *always* calls `triton_reshape_and_cache_flash`. Patch it to prefer `torch.ops._C_cache_ops.reshape_and_cache_flash` when `hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash")`, else fall back to Triton. Use a `hasattr` guard, **not** try/except-on-RuntimeError (cleaner, and matches your `_install_moe_C` guard style at cops.py:168).

**Validation:** unit-diff the native output vs the Triton path on a captured (key,value,slot_mapping) tuple for both auto and fp8 KV; then gemma-4 smoke (50 decode steps) checking logits drift <0.5% and no NaN.

**Effort:** 1–2 days. **Risk: LOW.** **Payoff:** +0.5–1.5% decode (one Triton launch → one HIP launch per step) **and** it unblocks K2 (native fp8-KV store) since `scaled_convert` is already in the kernel.

### S5 — generic native paged attention v1/v2 (do THIRD; **riskiest step**)

**Depends on:** F1 (softmax block-reduce finalize).

**Adapt/compile:**
- `vllm/csrc/attention/attention_kernels.cuh`, `paged_attention_v1.cu`, `paged_attention_v2.cu`. HEAD_SIZE=256 is already instantiated in both launchers. All shuffles route through `VLLM_SHFL_*` (wave32-clean). Include the F1 header for the finalize.
- `paged_attention_v1.cu` block-size switch instantiates {8,16,32}. **gemma-4 uses block_size=128** — this is the first thing to verify: either the kernel's per-token block loop maps a physical 32 to a logical 128 correctly, or you must add `case 64/128:` to the switch. **Read `attention_kernels.cuh` block-indexing (the `for block_idx ... += NUM_WARPS` loop) before compiling** — this is where head-256 wiring goes wrong.

**Create:**
- `win_attn_bindings.cu` → `TORCH_LIBRARY(_C, ...)` `.impl("paged_attention_v1"/"v2", torch::kCUDA, ...)` (these ops live in the `_C` namespace; `_custom_ops.py` already calls `torch.ops._C.paged_attention_v1`).
- New v1 backend wrapper `vllm/v1/attention/backends/generic_paged_attn.py` (~200–250 LOC, adapted from `rocm_attn.py`) — **Option A** from the S5 study. It splits KV via `PagedAttention.split_kv_cache`, then calls `ops.paged_attention_v1`. Keep it isolated rather than extending the `use_rocm_custom_paged_attention` gate — easier to A/B and can't regress the working Triton path.
- Build entry → `vllm_win_attn_C.pyd`; load it in `cops.py` like the others.

**The riskiest wiring — head-256 + sliding-window (flag):**
- **Sliding window is NOT in the C++ kernel.** It exists only in the Triton path (`kernel_paged_attention_2d.py` applies `S = where((ctx-off) < SLIDING_WINDOW, S, -inf)`). gemma-4's global-head layers are fine, but its **sliding head-256 layers will be numerically wrong** unless you port a token-offset mask into `attention_kernels.cuh` (~20 LOC, added to the causal mask at the QK stage). This is the single highest-risk item in the whole plan: it's a correctness bug that a smoke test can hide (small context < window) and only shows up at long context.
- Mitigation/sequencing: land S5 **global-head-only first** (route only non-sliding layers to the native backend, keep sliding layers on Triton), validate at long context, *then* add the sliding mask and flip the sliding layers over. Your attention is already split (KVarN global / TRITON_ATTN sliding), so this partial routing is natural.
- PARTITION_SIZE=512 in v2: for head-256 prefill the logits buffer can exceed 64KB LDS — verify before enabling v2 for prefill, or keep v2 decode-only (max_num_partitions=1, trivial).

**Validation:** unit vs Triton reference (tol ~1e-3 fp16) on head-256/block-128; then gemma-4 e2e at **long context (>1024)** specifically to catch sliding-mask errors; benchmark vs Triton baseline.

**Effort:** 3–4 days core (2–3 compile+test, 1 backend wiring) + ~1 day sliding-window port. **Risk: MEDIUM-HIGH**, concentrated entirely in the sliding-window correctness and block-128 indexing.

---

## 2. AITER — honest verdict

**Your "~1:1 portable" framing is too optimistic — gently, it's wrong for the parts that matter.** AITER's Python dispatch layer *accepts* gfx1100 (no compile-time `#if gfx942_only`), which is where the "1:1" impression comes from. But the ops worth having are gated by **hardware (MFMA) + missing tuning data + missing gfx1100 ASM blobs + a hard Windows `AITER_TRITON_ONLY` switch**, none of which a hipify pass fixes.

**(a) Portable now (non-matrix, hipify/Triton — but ~0 net gain):**
- Triton-based kernels (mha_triton, MoE routing/softmax/top-k, activations, cache ops, custom all-reduce non-MFMA paths). These run on gfx1100 via the Triton backend **at parity with what you already have** — vLLM already falls back to these same Triton paths. No speedup, not worth vendoring.

**(b) Retargetable to RDNA3 WMMA (real effort, low ROI for M=1):**
- **Opus GEMM (opus.hpp, rocWMMA-wrapped):** *compiles* on gfx1100 — proven by upstream PR #3236 (RDNA4 gfx1200/1201 WMMA w32/w64) and PR #3547 ("Gate opus fp8 code for gfx1100"). **But** PR #3547 only adds fp8 **stubs** (fp8 GEMM is non-functional), and there are **zero gfx1100 tuned configs** (the `*_tuned_gemm.csv` are keyed gfx942/950/1250) → dispatch falls to Triton speed anyway. Effort 2–3 wk, net ≈0 for M=1.
- **Fused MoE 2-stage / ck GEMM accumulation:** MFMA-locked; a WMMA rewrite is 3–8 wk **per kernel** plus re-tuning. At M=1 the expert GEMM is tiny (<1 ms/token on a 7900 XT); rewriting it is wasted effort.

**(c) Hard-blocked (skip entirely):**
- **ASM-tuned kernels** — MLA decode (17× on MI300), fmha_v3 fwd/bwd: distributed as per-arch `.co` VLIW blobs under `hsa/gfx942|950|1250` only. **No gfx1100 blob**, and regenerating needs AMD's MLI codegen + tuning expertise not in the public HIP SDK. Infeasible.
- **CK (Composable Kernel):** default instances filter on gfx9xx; no gfx11xx instantiation templates; CMake is Linux-only.
- **Windows:** `setup.py` sets `AITER_TRITON_ONLY=True` / `ENABLE_CK=False` whenever `sys.platform=="win32"`. Even a perfect gfx1100 CK port would be disabled on your OS.

**Specific kernels "worth it" for M=1 decode:** honestly, **none** at acceptable cost. The three you'd actually want (MLA, fmha_v3 paged-attn, fused MoE) are all MFMA/ASM-locked. Your **S5 generic paged-attention** is the same win as fmha_v3 for far less effort and is 100% RDNA3-native. **Recommendation: skip AITER**; RDNA3 enablement upstream is trickling in for RDNA4 (gfx1200) and CDNA, not gfx1100 — wait for it rather than spend 3–8 weeks for a 0–5% maybe.

*(Sources: ROCm/aiter `setup.py` Windows/`AITER_TRITON_ONLY` gate; `build_targets.py` GFX_MAP; PR #3236 RDNA4 WMMA; PR #3547 gfx1100 fp8 stubs; `hsa/` per-arch `.co` layout; CK gen_instances gfx9-only. Cross-checks your `docs/vllm-nvidia-vs-ours-gap-study.md` §X7.)*

---

## 3. Recommended sequencing + cumulative tok/s

Baseline: **56.8 tok/s** (gemma-4-26B-A4B, gfx1100).

| Step | What | Effort | Δ decode | Cumulative | Risk |
|------|------|--------|----------|-----------|------|
| **1. F1** | wave32 reduce/scan header | 0.5–1 d | ~0 (unblocker) | 56.8 | LOW |
| **2. S2** | native reshape_and_cache_flash (+unblocks K2) | 1–2 d | +0.5–1.5% | ~57–58 | LOW |
| **3. S5 (global-only)** | native paged-attn v1, non-sliding layers | 3–4 d | +2–5% | ~59–61 | MED |
| **4. S5 (sliding port)** | token-offset mask, flip sliding layers | ~1 d | +1–3% | ~60–63 | **HIGH** |
| — | **AITER** | 3–8 wk | 0–5% (maybe) | — | skip |

**Realistic cumulative: ~60–63 tok/s** after F1+S2+S5 (roughly +6–11% over 56.8). This is honest: your decode is memory-latency-bound (~150 serialized launches/step), so these kernel swaps mostly **cut launches and Triton overhead**, they don't change the DRAM-bandwidth ceiling. The bigger structural lever remains **launch-count reduction / graph capture coverage** (which S2 and a native MoE-decode GEMV feed into), not raw kernel FLOPs.

**Do in this order:** F1 (unblock) → S2 (prove kCUDA-native + v1 routing wiring cheaply, unblock K2) → S5 global-only (biggest single win) → S5 sliding (highest risk, gated behind long-context validation) → **do not** detour into AITER.

**Files to touch, at a glance:**
- Create: `csrc/rocm_wave32_reduce.cuh`, `win_cache_bindings.cu`, `win_attn_bindings.cu`, `vllm/v1/attention/backends/generic_paged_attn.py`, three build entries in `experiments/vllm_c_ext/build_c_ext.py`.
- Adapt: `csrc/cub_helpers.h`, `csrc/cache_kernels.cu` (hipMemcpy), `csrc/attention/*.cu[h]` (F1 include + sliding mask), `vllm/v1/attention/backends/triton_attn.py` (do_kv_cache_update guard), `windows_rocm_plugin/vllm_windows_rocm/cops.py` (three new `_load_native`-style loaders).
- **Keep `torch::kCUDA` in all `.impl(...)` calls — never `torch::kHIP` on this Windows build.**