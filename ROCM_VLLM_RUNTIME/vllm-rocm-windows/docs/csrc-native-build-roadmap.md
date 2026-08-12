# vLLM Windows ROCm Native Kernel Build Inventory — Synthesized Roadmap

This is the consolidated, prioritized build roadmap for the `_C` / `_rocm_C` / `_moe_C` native extensions on native Windows + AMD ROCm (gfx1100 / RDNA3, wave32, HIP SDK 7.2 / ROCm 7.13, MSVC + hipcc). Based on the five per-area inventories. Latency-bound M=1 decode of dense + MoE W4A16 is the target; **fusion kernels that cut serialized launches and the `rocm/` skinny GEMMs are the real lever.**

---

## 1. Master File Inventory

Verdicts: **hipify** = compiles as-is via the proven regex-hipify recipe · **adapt** = minor edits · **rewrite** = substantial · **skip** = not needed for single-GPU RDNA3 W4A16 decode.

### Core (`csrc/`) — foundation + fusion kernels

| File | Purpose | Verdict | Perf | Blocker |
|---|---|---|---|---|
| cuda_compat.h | warp-size/shuffle/sync/attr macros | hipify | crit-support | none (already ROCm-aware) |
| dispatch_utils.h | type-dispatch macros, dual-FP8 | hipify | crit-support | none |
| type_convert.cuh | HIP/CUDA type bridges | hipify | crit-support | none |
| cuda_utils.h | ceil_div, error checks | hipify | low | none |
| ops.h / torch_bindings.cpp | op decls + registration | hipify | crit-support | USE_ROCM guards already correct |
| pos_encoding_kernels.cu | RoPE | hipify | med | pure math; already have rotary in .pyd |
| activation_kernels.cu | silu/gelu + gating fused | adapt | HIGH | `getCurrentDeviceProperties()->major`, CUDA_VERSION checks; partly in .pyd |
| **layernorm_kernels.cu** | RMS norm + fused add+norm | **rewrite** | **VERY HIGH** | `cub::BlockReduce` → rocPRIM or hand-rolled warp-reduce |
| **layernorm_quant_kernels.cu** | norm + FP8 quant fusion | **rewrite** | **VERY HIGH** | same CUB dep; verify scaled_fp8_conversion |
| **cache_kernels.cu** | reshape_and_cache, copy, concat | **adapt** | **VERY HIGH** | warp=32 hardcode (OK on RDNA3), cudaMemcpy→hipMemcpy |
| **fused_qknorm_rope_kernel.cu** | QK-norm + RoPE fused (3→1 launch) | **adapt** | **VERY HIGH** | warp macros handled by cuda_compat; warp=32 OK |
| cache_kernels_fused.cu | MLA fused rope+cache | hipify | med | skip unless MLA model; ROCm path present |
| sampler.cu / topk.cu | sampling | skip | low | Triton fallback fine |
| custom_all_reduce.cu / custom_quickreduce.cu | multi-GPU comms | skip | — | single-GPU |

### `csrc/rocm/` — the skinny-GEMM + attention lever

| File | Purpose | Verdict | Perf | Blocker |
|---|---|---|---|---|
| **skinny_gemms.cu** | LLMM1 + wvSplitK M=1/skinny GEMV | **adapt** | **HIGH** | wave32 branching already present; wvSplitKrc gfx950-only (skip), wvSplitKQ FP8 gfx12-only (stub) |
| attention.cu | ROCm paged attn (MFMA) | rewrite | HIGH | hardcoded warp=64 `__shfl_*`, MFMA16 gfx9-only, no gfx11 path |
| ops.h / torch_bindings.cpp | decls + registration | hipify | support | none |

### `csrc/attention/` — portable paged attention (the better attention bet)

| File | Purpose | Verdict | Perf | Blocker |
|---|---|---|---|---|
| **attention_kernels.cuh** | core paged-attn v1/v2 (fused QK+softmax+V) | hipify | **HIGH** | VLLM_SHFL macros; wave32==warp32 so masks match |
| paged_attention_v1.cu / v2.cu | decode launchers | hipify | HIGH | cudaStream_t maps in ATen |
| attention_generic/utils.cuh | vector ops, QK reduce | hipify | high | none |
| dtype_float16.cuh | FP16 vec ops + asm | hipify | HIGH | inline asm already has ROCm branch |
| dtype_bfloat16.cuh | BF16 vec ops | hipify | HIGH | CUDA_ARCH gate OK on gfx1100 |
| dtype_float32/fp8.cuh | FP32/FP8 types | hipify | low/med | none |
| merge_attn_states.cu | split-KV merge | skip | low | multi-GPU/prefill |
| vertical_slash_index.cu | sparse index | skip | low | cudaSetDevice→hipSetDevice if ever enabled |
| mla/sm100_cutlass_mla_kernel.cu | MLA (SM100) | skip | — | CUTLASS, NVIDIA-only |

> **Key cross-area insight:** `csrc/attention/attention_kernels.cuh` (generic, wave32-clean, hipify-able) is a **much cheaper attention win** than rewriting `csrc/rocm/attention.cu` (wave64 + MFMA, gfx9-specific). Port the generic one; treat rocm/attention.cu as a later gfx1100-specific optimization, not the baseline.

### `csrc/moe/` — MoE routing + expert GEMM

| File | Purpose | Verdict | Perf | Blocker |
|---|---|---|---|---|
| **moe_wna16.cu** (+ moe_wna16_utils.h) | W4A16 expert GEMM | **adapt** | **HIGH** | half2 swizzle verify, CUDA_ARCH→gfx guard, PTX lop3/prmt in utils |
| **topk_softmax_kernels.cu** | fused top-k + softmax routing | **adapt** | **HIGH** | shuffle width, CUB→rocPRIM, warp-size |
| moe_align_block_size.cu | token→expert permute | adapt | med | CUB BlockScan→rocPRIM |
| torch_bindings.cpp | registration | hipify | support | USE_ROCM guards correct |
| router_gemm.cu | router GEMM (cuBLAS) | adapt | med | cuBLAS→rocBLAS (or keep Triton; M=1 tiny) |
| grouped_topk_kernels.cu | grouped routing (DeepSeek) | rewrite/skip | med | cooperative_groups, ballot, SM90 grid-dep — skip (Gemma isn't grouped) |
| marlin_moe_wna16/* | Marlin MoE | rewrite/skip | — | PTX lop3/prmt, tensor-core tiling |
| gpt_oss_router_gemm.cu | SM90 router | skip | — | CUTensorMap, cudaLaunchKernelEx |
| moe_permute_unpermute_op.cu | multi-GPU expert-parallel | skip | — | CUTLASS, CUDA12 gates |
| mxfp8_moe/* | MxFP8 (Blackwell) | skip | — | CUTLASS, SM100 |
| dynamic_4bit_int_moe_cpu.cpp | ARM64 CPU | skip | — | AArch64-only |

### `csrc/quantization/` — the W4A16 decode core

| File/area | Purpose | Verdict | Perf | Blocker |
|---|---|---|---|---|
| **awq/gemm_kernels.cu** | AWQ W4A16 GEMM | **rewrite** | **HIGH** | ldmatrix + mma.sync PTX (tensor cores); need hand-rolled wave32 M=1 GEMV |
| gptq/q_gemm.cu | GPTQ W4A16 + hipBLAS fallback | hipify | med | already has `#if USE_ROCM` + hipBLAS path |
| activation_kernels.cu (quant) | fused silu+mul+quant | hipify | med | ROCm skeleton present; fill wave32 shuffles |
| fused_layernorm_dynamic_per_token_quant.cu | fused LN+per-token quant | hipify | med | verify no warp-only ops |
| w8a8/fp8/common.cu, int8/scaled_quant.cu | FP8/INT8 quant | hipify | low | portable; only if W8A8 needed |
| marlin/dequant.h | INT4/FP4 dequant helpers | adapt | med | PTX lop3 → HIP bit ops |
| marlin/marlin.cu, marlin_template.h | Marlin GEMM | rewrite/skip | med | ldmatrix PTX; confirm model format first |
| marlin/*_repack.cu | offline repack | skip | low | utility |
| gptq_allspark/* | AllSpark W8A16 | skip | low | Ampere MMA |
| w8a8/cutlass/* (18), cutlass_w4a8/*, fp4/nvfp4_*, machete/* | CUTLASS/FP4 | skip | — | **CUTLASS not on Win ROCm** |
| gguf/gguf_kernel.cu | GGUF ops | skip | — | llama.cpp format |
| hadamard/* | Hadamard transform | skip | low | utility |

---

## 2. Incremental Build Plan (ranked by latency payoff per unit effort)

Your own perf notes say hand-rolling faster individual GEMVs "gave ~nothing" — the win is **cutting the ~150 serialized launches/step**. The plan is biased accordingly: fusion first, then skinny GEMMs, defer per-op micro-opt.

### Tier 0 — Foundation (do first, ~1 day, near-zero risk)
Enables everything else; all `hipify`.
1. `cuda_compat.h`, `dispatch_utils.h`, `type_convert.cuh`, `cuda_utils.h`, `ops.h`, `torch_bindings.cpp`.
2. Verify the existing `.pyd` ops (silu_and_mul, gelu*, rms_norm, fused_add_rms_norm, rotary, weak_ref) still register cleanly under the unified target.

**Effort: ~1 day. Payoff: unblocks the rest.**

### Tier 1 — Fusion kernels that cut launches (highest payoff/effort)
These directly reduce the serialized-launch count that dominates decode.
1. **fused_qknorm_rope_kernel.cu** (`adapt`) — collapses qk_norm + rope + store (3→1) every step. warp macros already routed through cuda_compat. **Best single payoff.**
2. **cache_kernels.cu** (`adapt`) — `reshape_and_cache` runs per-token per-step; warp=32 hardcode is correct on RDNA3, only cudaMemcpy→hipMemcpy + vec-utils check.
3. **layernorm_kernels.cu** + **layernorm_quant_kernels.cu** (`rewrite`) — the only real work here is replacing `cub::BlockReduce`. Hand-roll a wave32 block-reduce (shared-mem tree, single warp finalize) once, reuse in both. ~2–3h for the reducer, then both files fall in.
4. **activation_kernels.cu** completeness (`adapt`) — finish fatrelu/swigluoai variants; swap `getCurrentDeviceProperties()->major` for a hip query.

**Effort: ~3–5 days. Payoff: removes the largest chunk of per-step launches. This is the core of the whole project.**

### Tier 2 — Skinny GEMMs + MoE routing (the M=1 GEMV lever)
1. **rocm/skinny_gemms.cu** (`adapt`) — LLMM1 + wvSplitK compile with the wave32 branching already in the file. **Skip wvSplitKrc (gfx950) and wvSplitKQ (gfx12 FP8)** — stub them out; not needed for RDNA3 W4A16. This is the intended replacement for `rocm_unquantized_gemm`'s wvSplitK/LLMM1 path you stubbed off.
2. **moe_wna16.cu** (`adapt`) — the W4A16 expert GEMM for the MoE path; verify half2 swizzle + guard CUDA_ARCH, port the lop3/prmt in utils to HIP bit-ops.
3. **topk_softmax_kernels.cu** (`adapt`) — fused routing every MoE step; CUB→rocPRIM + shuffle width.
4. **moe_align_block_size.cu** (`adapt`) — CUB BlockScan→rocPRIM; med priority (amortized).

**Effort: ~4–7 days. Payoff: HIGH for both dense (skinny GEMM) and MoE (expert GEMM + routing).**

### Tier 3 — Native paged attention (portable path)
1. **attention/attention_kernels.cuh + paged_attention_v1.cu / v2.cu + dtype_float16/bfloat16.cuh + generic/utils** (`hipify`) — wave32-clean generic paged attention; fuses QK+softmax+V. Cheaper and safer than rocm/attention.cu.

**Effort: ~2–4 days. Payoff: MED-HIGH (removes attention Triton fallback launches).**

### Tier 4 — AWQ W4A16 GEMM rewrite (deferred, biggest single rewrite)
1. **awq/gemm_kernels.cu** (`rewrite`) — tensor-core ldmatrix/mma.sync has no RDNA3 equivalent; needs a hand-rolled wave32 M=1 dequant-GEMV. You already have a working dequant-GEMV fallback (50.4 tok/s). Only invest here if profiling shows the AWQ GEMM is still the top cost after Tiers 1–3.

**Effort: ~1–2 weeks. Payoff: model-dependent; you already have a decent fallback, so defer.**

### Defer
- rocm/attention.cu (wave64+MFMA gfx9 rewrite) — use generic attention instead.
- marlin/* and marlin_moe_wna16/* — only if a target model is actually Marlin-packed.
- router_gemm.cu cuBLAS→rocBLAS — M=1 router is tiny; keep Triton until it shows up in a profile.
- grouped_topk_kernels.cu — only for DeepSeek-style grouped routing.

### Skip entirely
All CUTLASS-based (w8a8/cutlass, cutlass_w4a8, fp4/nvfp4, machete, mxfp8_moe, mla/sm100), gguf, hadamard, gptq_allspark, sampler.cu, custom_all_reduce/quickreduce, moe_permute_unpermute, dynamic_4bit_int_moe_cpu, merge_attn_states, vertical_slash_index.

---

## 3. Risky External Deps — rewrite vs skip

| Dep | Win ROCm status | Decision |
|---|---|---|
| **CUTLASS** | Not available (CUDA PTX/SASS only; no HIP port) | **Skip all CUTLASS sources.** For W4A16, the non-CUTLASS paths (moe_wna16, gptq q_gemm, hand-rolled skinny GEMM) cover the decode path. No rewrite needed — CUTLASS kernels are NVIDIA tensor-core specific and wouldn't map to RDNA3 anyway. |
| **CUB / Thrust** | Not available (NVIDIA-only) | **Rewrite dependency-free.** This is the one that blocks VERY-HIGH files (layernorm, layernorm_quant) and HIGH MoE files (topk_softmax, align). Don't pull in rocPRIM as a new dep — hand-roll one wave32 block-reduce + a small block-scan helper and reuse everywhere. Quality/perf fully preserved (a block-reduce is trivial to match). ~half a day total. |
| **composable_kernel (CK)** | Not in vLLM tree; Win status unknown/likely missing | **Skip** — no vLLM source in your target set requires it. |
| **hipBLASLt** | Shipped in HIP SDK 7.2, but vLLM doesn't call it | Not needed now. Optional future lever for larger-M prefill GEMMs; ignore for decode. |
| **rocBLAS / hipBLAS** | Shipped + proven working | **Use as-is.** Already linked in the proven recipe; gptq q_gemm's hipBLAS fallback works on gfx1100. |
| **PTX inline asm** (ldmatrix, mma.sync, lop3, prmt) | No RDNA3 equivalent | ldmatrix/mma.sync (awq, marlin) → **rewrite as wave32 GEMV** or skip the file. lop3/prmt (dequant.h, moe_wna16_utils) → **rewrite as HIP bit-ops** (v_bfe/v_perm or plain C bit-math); cheap and quality-preserving. |

**Bottom line:** the only external dep you must *replace* is **CUB**, and the right move is a small hand-rolled wave32 reduce/scan (not adopting rocPRIM). Everything else is either shipped (rocBLAS/hipBLAS) or skippable (CUTLASS/CK).

---

## 4. What We Already Have vs Gaps

### Have (proven working)
- **Build recipe** (`experiments/vllm_c_ext/`): regex-hipify (torch's `RE_PYTORCH_PREPROCESSOR` + `PYTORCH_MAP`, bypassing Windows-broken hipify orchestrator) + ATen/c10 `cuda→hip` shim headers + `torch.utils.cpp_extension.load()` with `--rocm-device-lib-path`, `-DUSE_ROCM=1`, `-DTORCH_HIP_VERSION=0`, undef HALF guards, link rocblas/hipblas/amdhip64. **This is the load-bearing asset — reuse it, don't fight CMake.**
- **vllm_win_C.pyd**: silu_and_mul, gelu_and_mul, gelu_tanh_and_mul, rms_norm, fused_add_rms_norm, rotary_embedding, weak_ref_tensor (+ gptq_gemm/gptq_shuffle bindings).
- **vllm_win_moe_C.pyd**: topk_softmax, moe_sum, moe_align_block_size, batched_moe_align_block_size bindings.
- Working dequant-GEMV fallback path (gemma-4-26B at 50.4 tok/s).

### Gaps (ranked by impact)
1. **No fused qk-norm+rope, no native reshape_and_cache, no fused norm(+quant)** — the biggest launch-count reductions are all still on Triton/Python. → Tier 1.
2. **skinny_gemms.cu (LLMM1/wvSplitK) not built** — `rocm_unquantized_gemm`'s intended fast path is stubbed off; dense M=1 GEMVs run fallbacks. → Tier 2.
3. **moe_wna16 expert GEMM not native** — MoE experts run Triton/dequant fallback. → Tier 2.
4. **topk_softmax is bound but the kernel isn't the fused ROCm-optimized one** — verify the registered kernel is real vs a stub. → Tier 2.
5. **No native paged attention** — attention on Triton fallback. → Tier 3.
6. **AWQ GEMM** — only the dequant fallback exists; no native tensor-equivalent. → Tier 4 (acceptable gap given current 50.4 tok/s).

### Honest effort estimate
- **Tier 0+1 (foundation + fusion): ~1 week.** This is where most of the decode latency win lives and it's mostly `adapt` + one CUB-reducer rewrite. Do this before anything else.
- **Tier 2 (skinny GEMM + MoE): ~1–1.5 weeks.** Mostly `adapt`; the wave32 branching in skinny_gemms already exists.
- **Tier 3 (attention): ~0.5–1 week** via the generic (non-rocm/) kernels.
- **Tier 4 (AWQ rewrite): 1–2 weeks**, deferred until profiling justifies it.

**Total to a strong native decode build: ~3–4 weeks**, front-loaded so the highest-payoff fusion kernels land in week one. Skip CUTLASS entirely, replace only CUB (hand-rolled), and lean on the already-proven `vllm_c_ext` recipe rather than the upstream CMake/hipify orchestrator.