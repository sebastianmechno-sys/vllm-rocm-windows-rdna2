"""Skinny bf16/fp16 M=1 GEMV for the DENSE (unquantized) linears on RDNA3.

Profiling gemma-4-26B-A4B decode showed the per-layer DENSE MLP (bf16, unquantized shared expert,
gate/up/down at intermediate 2112) is ~43% of the per-token weight bytes -- the single biggest
memory consumer. On ROCm it goes through rocm_unquantized_gemm -> wvSplitK (the skinny M=1 GEMM,
ABSENT on our Windows build) -> torch.nn.functional.linear (rocBLAS), which is GEMM-tuned and
under-utilises the 84 CUs at M=1 for these small (2112/2816-row) shapes (rocBLAS ~330 GB/s cached).

This is a small-tile GEMV (few output rows per program -> many programs -> the CUs stay busy),
streaming the weight once. Patches UnquantizedLinearMethod.apply for the M==1 decode step only;
everything else (prefill, M>1) falls through to the original path. Enable VLLM_WIN_BF16_GEMV=1.
"""
import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _bf16_gemv_kernel(x_ptr, w_ptr, b_ptr, o_ptr, K, N,
                      HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[n] = sum_k w[n,k]*x[k] (+ bias[n]).  w is [N, K] row-major; x is [K].
    # One program handles BLOCK_N contiguous output rows; loops the K reduction. Small BLOCK_N ->
    # many programs -> full CU occupancy for the small dense-MLP shapes.
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    acc = tl.zeros((BLOCK_N,), tl.float32)
    wrow = w_ptr + n[:, None] * K
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        kmask = kk < K
        x = tl.load(x_ptr + kk, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(wrow + kk[None, :], mask=nmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=nmask, other=0.0).to(tl.float32)
    tl.store(o_ptr + n, acc.to(o_ptr.type.element_ty), mask=nmask)


def bf16_gemv(x_row, weight, bias):
    # x_row [K] contiguous; weight [N, K]; bias [N] or None -> out [N]
    N, K = weight.shape
    o = torch.empty(N, dtype=x_row.dtype, device=x_row.device)
    grid = lambda m: (triton.cdiv(N, m["BLOCK_N"]),)
    BN = 32 if N >= 8192 else 16
    BK = 1024 if K >= 2048 else 256
    NW = 8 if N >= 8192 else 4
    _bf16_gemv_kernel[grid](
        x_row, weight, bias if bias is not None else x_row, o, K, N,
        HAS_BIAS=bias is not None, BLOCK_N=BN, BLOCK_K=BK, num_warps=NW)
    return o


_PATCHED = False


def patch_unquantized_linear() -> None:
    global _PATCHED
    if _PATCHED or os.environ.get("VLLM_WIN_BF16_GEMV", "0") != "1":
        return
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )
    except Exception as e:  # noqa: BLE001
        print("vllm-win bf16_gemv patch warning:", repr(e))
        return
    validate = os.environ.get("VLLM_WIN_BF16_GEMV_VALIDATE", "0") == "1"
    _orig = UnquantizedLinearMethod.apply
    _orig_emb = UnquantizedEmbeddingMethod.apply

    def _gemv_apply(apply_orig, self, layer, x, bias=None):
        w = layer.weight
        if (x.dim() >= 2 and x.shape[:-1].numel() == 1 and w.dim() == 2
                and w.dtype in (torch.bfloat16, torch.float16)
                and x.shape[-1] == w.shape[1]):
            # Tied lm_head (cyankiwi) stores BF16 while activations are fp16: cast the weight
            # once and cache it on the layer (also halves the streamed bytes per token) instead
            # of converting every decode step.
            if w.dtype != x.dtype:
                w = getattr(layer, "_gemv_w_cast", None)
                if w is None:
                    w = layer.weight.to(x.dtype)
                    layer._gemv_w_cast = w
            xr = x.reshape(-1)  # [K]
            y = bf16_gemv(xr.contiguous(), w, bias)
            out = y.reshape(*x.shape[:-1], w.shape[0])
            if validate:
                ref = apply_orig(self, layer, x, bias)
                rel = (out.float() - ref.float()).abs().max().item() / (
                    ref.float().abs().max().item() + 1e-6)
                print(f"BF16_GEMV_VALIDATE N={w.shape[0]} K={w.shape[1]} rel={rel:.3e}")
                return ref
            return out
        return apply_orig(self, layer, x, bias)

    def apply(self, layer, x, bias=None):
        return _gemv_apply(_orig, self, layer, x, bias)

    def emb_apply(self, layer, x, bias=None):
        return _gemv_apply(_orig_emb, self, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    UnquantizedEmbeddingMethod.apply = emb_apply
    _PATCHED = True
    print("vllm-win: patched UnquantizedLinearMethod/EmbeddingMethod.apply (M=1 skinny GEMV)")
