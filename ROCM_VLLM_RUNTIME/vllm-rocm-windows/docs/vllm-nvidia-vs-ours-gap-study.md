Both spot-checks confirm the surveys' claims: `fused_qknorm_rope_kernel.cu` exists, the full `spec_decode/` framework tree is present, and the roadmap's verdicts match. I have enough to synthesize. Here is the master study document.

---

# vLLM NVIDIA-Linux vs Our Native Windows-ROCm Port (gfx1100 / RDNA3, wave32) — Master Gap Study

**Baseline:** gemma-4-26B-A4B @ **56.8 tok/s** decode, memory-latency-bound, no matrix cores, ~800 GB/s DRAM, 24 GB. Method = regex-hipify csrc + ATen/c10 cuda→hip shim + `cpp_extension.load`; CUB → hand-rolled wave32 reduce/scan; skip CUTLASS/MFMA/PTX-mma.

---

## 1. MASTER GAP TABLE

Keyword codes: **V**=Velocità (speed), **P**=Precisione, **M**=VRAM. Portability: **hipify-1:1** / **adapt** (minor edits) / **rewrite** (dependency-free, substantial) / **infeasible**.

| # | Feature / op | NVIDIA-Linux vLLM has | Our port has | Keyword | RDNA3 fit? | Portability | Effort |
|---|---|---|---|---|---|---|---|
| **FOUNDATION** ||||||||
| F1 | Wave32 block-reduce/scan (replaces `cub::BlockReduce`/`BlockScan`) | CUB (CUDA toolkit) | ✗ (Triton fallbacks) | V,P | yes | **rewrite** (~80 LOC, reusable) | 0.5–1 d |
| **FUSION (SPEED)** ||||||||
| S1 | `fused_qknorm_rope_kernel.cu` (3→1 launch) | ✓ native | ✗ (Triton, 3 launches) | V | yes (warp32 hardcode OK) | **adapt** | 2–3 d* |
| S2 | `cache_kernels.cu` reshape_and_cache | ✓ native | ✗ (Triton) | V,M | yes | **adapt** (cudaMemcpy→hip) | 1–2 d |
| S3 | `layernorm_kernels.cu` (rms/fused_add already in .pyd via torch fallback; native+quant not) | ✓ native | partial (.pyd torch fallback) | V,P | yes | **rewrite** (CUB dep) | 1 d after F1 |
| S4 | `activation_kernels.cu` full native (silu/gelu in .pyd already) | ✓ native | partial | V | yes | **adapt** (device-props query) | 1 d |
| S5 | `attention_kernels.cuh` + paged_attention_v1/v2 (generic, wave32-clean) | ✓ native | ✗ (TRITON_ATTN) | V | yes (VLLM_SHFL routed) | **hipify** (+F1 for split-KV reduce) | 3–4 d |
| **QUANT / MoE (SPEED + PRECISION)** ||||||||
| Q1 | `moe/topk_softmax_kernels.cu` fused route | ✓ native | ✗ (Triton) | V,P | yes | **adapt** (CUB→F1) | 1–2 d after F1 |
| Q2 | `moe/moe_align_block_size.cu` | ✓ native | ✗ (Triton) | V | yes | **adapt** (BlockScan→F1) | 1 d after F1 |
| Q3 | `moe/moe_wna16.cu` W4A16 expert GEMM | ✓ native | ✗ (Triton dequant + GEMV) | V | conditional (half2 swizzle → wave32 retune) | **adapt/rewrite** | 4–7 d |
| Q4 | `marlin/dequant.h` int4→fp16 bit-tricks | ✓ (PTX lop3/prmt) | ✗ (scalar) | V | yes | **adapt** (lop3→HIP v_perm/C bit-math) | 4 h |
| Q5 | `w8a8/fp8/common.cu` FP8 quant (per-token/channel/block) | ✓ native | ✗ (Triton) | P,M | yes (warp-reduce, no CUB) | **hipify** | 1–2 d |
| Q6 | `fused_layernorm_dynamic_per_token_quant.cu` | ✓ native | ✗ | V,P,M | yes | **rewrite** (CUB→F1) | 2–3 d after F1 |
| Q7 | `int8/scaled_quant.cu` | ✓ native | ✗ | P | yes | **hipify** | 1 d |
| **KV / VRAM** ||||||||
| K1 | KVarN K4V2/K4V4 per-tile | ✓ (also upstream now) | ✓ **done** | M,P | yes | done | — |
| K2 | Native fp8 KV store (reshape_and_cache + `scaled_convert`) | ✓ native | ✗ (Triton) — scale-load Python OK | M,P | yes (amd/quant_utils has USE_ROCM) | **hipify** (dep: S2) | 2–3 d |
| K3 | `rocm/skinny_gemms.cu` LLMM1 + wvSplitK (dense M=1) | ✓ (gfx9/gfx12 variants) | partial (.pyd built, not in prod) | V | yes (wave32 branch present) | **adapt** (stub gfx950/gfx12 paths) | 1–2 d |
| K4 | Prefix caching (hash block reuse) | ✓ Python | ✓ (inherited) | V,M | yes | done | — |
| K5 | Sliding window / block-sparse mask | ✓ (in paged_attn IS_BLOCK_SPARSE + Triton) | ✓ (Triton mask) | M | yes | done (native comes free w/ S5) | — |
| K6 | SimpleCPUOffload (CPU KV spill) | ✓ Python + async H2D | ? untested | M | yes | **adapt** (test only) | 1 d |
| K7 | Chunked prefill | ✓ | ✓ (Triton) | M | yes | done | — |
| **DECODE ALGOS (framework, no kernels)** ||||||||
| D1 | ngram speculative (Numba CPU) | ✓ `ngram_proposer.py` | ✗ (no spec framework) | V | yes (CPU) | **adapt** (port framework + plumbing) | 1 d (+2–3 d framework) |
| D2 | EAGLE3 + tree attention (Triton) | ✓ | ✗ | V | yes (Triton) | **adapt** | 2–3 d |
| D3 | Medusa heads | ✓ | ✗ | V | yes | **adapt** | 1.5 d |
| D4 | MLP speculator | ✓ | ✗ | V | yes | **adapt** | 1 d |
| D5 | FULL cudagraph (decode) | ✓ | ✓ **done** (FULL_DECODE_ONLY) | V | yes | done | — |
| D6 | Draft-model spec | ✓ | ✗ | V | **no** (VRAM: +10–12 GB on 24 GB card) | infeasible-here | — |
| **INFEASIBLE (tensor-core / sm90 / cubin)** ||||||||
| X1 | AWQ `gemm_kernels.cu` | ✓ (ldmatrix+mma.sync) | dequant-GEMV fallback | — | no | **infeasible** | — |
| X2 | Marlin / Machete GEMM | ✓ CUTLASS/PTX | ✗ | — | no | **infeasible** | — |
| X3 | W8A8 CUTLASS, cutlass_w4a8 | ✓ | ✗ | — | no (sm90/100/120) | **infeasible** | — |
| X4 | FP4 / NvFP4 / MxFP8 scaled MM | ✓ CUTLASS sm120 | ✗ | — | no | **infeasible** | — |
| X5 | FlashAttention 2/3/4 | ✓ (HMMA/TMA/TMEM) | ✗ | — | no | **infeasible** | — |
| X6 | FlashInfer (TRTLLM) | ✓ cubin | ✗ | — | no | **infeasible** | — |
| X7 | AITER / Composable Kernel | ✓ (MI3xx gfx942/950) | ✗ | — | **no — gfx1100 unsupported upstream** | **infeasible** | — |
| X8 | `rocm/attention.cu` (MFMA paged attn) | ✓ (gfx9) | ✗ | — | no (MFMA/wave64) | **infeasible** (use S5) | — |
| X9 | gptq_allspark W8A16 | ✓ (Ampere MMA) | ✗ | — | no | **infeasible** | — |

\* S1 is `adapt` at compile level but the surveys flag it already builds+correct yet measured **−1.8%** on your model — see §3.

---

## 2. RANKED "PORT NEXT" — highest payoff-per-effort, genuinely wave32-portable

### Prerequisite (do first — unblocks everything below)
**F1 — Hand-rolled wave32 block-reduce + block-scan** (0.5–1 d). ~80 LOC header (`rocm_wave32_reduce.cuh`), reused by S3, Q1, Q2, Q6, and S5's split-KV finalize. Numerically matches CUB. This is the single highest-leverage item — skip it and half the SPEED list stays blocked.

### Top picks for SPEED (Velocità)
1. **S2 reshape_and_cache** (1–2 d, adapt) — per-token, no CUB, unblocks native fp8 KV (K2). Clean win.
2. **S5 generic paged attention** (3–4 d, hipify + F1) — replaces TRITON_ATTN in the decode hot path; surveys estimate +10–25% decode. **Biggest kernel-level SPEED lever.** Note the gemma-4 constraint: head=256 + sliding window must be covered by template instantiation — validate head_size ∈ {128,256} and IS_BLOCK_SPARSE paths before trusting the number.
3. **K3 skinny_gemms** (1–2 d, adapt) — .pyd already built; wire LLMM1/wvSplitK into dense M=1, stub gfx950/gfx12. Low risk, +1–2 tok/s.
4. **Q1+Q2 topk_softmax + moe_align** (2–3 d after F1, adapt) — MoE-only; gemma-4 is MoE so this applies. +3–5% MoE decode.
5. **D1 ngram speculative** (1 d + one-time 2–3 d framework, adapt) — pure framework/CPU, +8–12%, zero kernel risk. **Best non-kernel SPEED bet.** D2 EAGLE3 (+15%) stacks after the framework exists.
6. **Q4 marlin/dequant.h** (4 h, adapt) — cheap, +1–2% dequant, reusable by GPTQ/AWQ paths.

### Top picks for PRECISION (Precisione)
1. **K1 KVarN K4V4 retune** (2 d) — you already own the kernel; tightening Sinkhorn iters buys +1–2% accuracy at 4× cache reduction. Highest precision ROI.
2. **Q5 FP8 dynamic per-token quant** (1–2 d, hipify) — per-token scales vs per-tensor, +1–2% on W8A8 models. No CUB, warp-reduce only.
3. **Q6 fused norm+dynamic-per-token-quant** (2–3 d after F1) — precision + a launch cut. Only matters once you run a W8A8 model; defer for pure W4A16 gemma.

### Top picks for VRAM
1. **K1 KVarN** — already done, ~45–50% KV savings; the K4V4 retune above is the only remaining lever here.
2. **K2 native fp8 KV store** (2–3 d, hipify; dep S2) — 2× vs fp16, complements KVarN for outlier-heavy models. Scale-loading Python already works.
3. **K6 CPU offload** (1 d, test) — unbounded sequence length; likely already works, needs validation only.

---

## 3. NOT WORTH IT / INFEASIBLE ON RDNA3 — with reasons

| Item | Why not |
|---|---|
| **AWQ/Marlin/Machete GEMM** (X1,X2) | ldmatrix + mma.sync PTX; no wave32/scalar equivalent. Dequant-GEMV fallback already covers M=1 decode. A from-scratch wave32 int4 GEMV would beat conch by only ~10–15% for 2–3 weeks work — not worth it at 56.8 tok/s. |
| **CUTLASS W8A8 / cutlass_w4a8 / FP4 / NvFP4 / MxFP8** (X3,X4) | CUTLASS + sm90/100/120 tensor cores. No RDNA3 rocWMMA on gfx1100. INT4+FP8+KVarN already cover the precision space. |
| **FlashAttention 2/3/4** (X5) | HMMA (SM70+), Hopper TMA, Blackwell TMEM. ROCm FA variant is AITER/gfx9-only. Generic paged attn (S5) is the correct bet. |
| **FlashInfer** (X6) | Cubin/CUDA-only, no public ROCm source. Porting = months. Triton fallback stands. |
| **AITER / Composable Kernel** (X7) | Upstream supports MI300/MI325 (gfx942/950) only; **gfx1100 explicitly unsupported**. Blocked until AMD ships gfx11 kernels. |
| **rocm/attention.cu** (X8) | Hardcoded wave64 + MFMA16, gfx9-only. Generic `attention_kernels.cuh` (S5) is cheaper and actually runs on RDNA3. |
| **gptq_allspark** (X9) | Ampere MMA, nvcc-only. Use standard GPTQ + hipBLAS. |
| **Draft-model speculative** (D6) | Needs +10–12 GB for a drafter; target already uses ~18 GB of 24 GB. VRAM-infeasible (ngram/EAGLE are the right spec choices). |
| **PIECEWISE cudagraph** | Complex on ROCm; FULL_DECODE_ONLY already captured and sufficient for single-stream M=1. |
| **fused_qknorm_rope (S1)** | Already found dead-end: builds + numerically correct but **−1.8%** measured on gemma-4. Keep the code, don't ship it; revisit only if profiling changes. |
| **moe_wna16 as-is (Q3)** | Throughput/sorted-token kernel, half2 swizzle tuned for wave64-style packing; needs real wave32 retune (4–7 d) and only helps if it beats your existing Triton dequant+GEMV at M=1 — uncertain. Medium-priority, not a quick win. |

---

## 4. Strategic read — where's the ceiling, which keyword has headroom

You're memory-latency-bound with no matrix cores, so the ceiling is set by DRAM round-trips per token, not FLOPs — llama.cpp at ~85 tok/s on the same silicon is the practical near-term target (1.5× away), not the NVIDIA tensor-core numbers, which are structurally unreachable (every high-value NVIDIA GEMM is CUTLASS/MFMA-locked and correctly excluded). The remaining kernel headroom is almost entirely **SPEED via launch-count reduction**: the decode step still fires ~150+ serialized Triton launches, and the portable fusion stack — F1 (wave32 reduce) → S2 (reshape_and_cache) → S5 (generic paged attention) → Q1/Q2 (MoE routing) — is realistically worth +15–25% and lands you around 65–70 tok/s over ~2–3 weeks with zero precision risk. Beyond kernels, the largest untapped SPEED lever is architectural, not arithmetic: **speculative decoding (ngram then EAGLE3)** is pure framework/Triton, RDNA3-neutral, and can compound another +20–27% because it attacks the memory-latency bottleneck directly by amortizing DRAM traffic across accepted draft tokens — this is the one place a 24 GB gfx1100 can plausibly pass 80 tok/s. **PRECISION is effectively solved** (KVarN owns it; only a cheap K4V4 retune remains) and **VRAM is comfortable** (KVarN + optional fp8 KV + CPU offload leave you slack on 24 GB). Net: pour effort into the fusion/paged-attention chain and the speculative-decode framework; treat everything tensor-core-shaped as permanently out of scope.

**Files to touch first (all absolute):**
- Create: `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc\rocm_wave32_reduce.cuh`
- `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc\cache_kernels.cu`
- `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc\attention\attention_kernels.cuh` (+ `paged_attention_v1.cu`, `v2.cu`)
- `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc\moe\topk_softmax_kernels.cu`, `moe_align_block_size.cu`
- `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc\quantization\marlin\dequant.h`
- `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc\rocm\skinny_gemms.cu`
- Framework: `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\vllm\v1\spec_decode\` (ngram_proposer.py, eagle.py, metadata.py), `config\speculative.py`
- Roadmap reference: `C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\docs\csrc-native-build-roadmap.md`