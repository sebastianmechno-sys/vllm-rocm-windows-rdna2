"""Build vLLM's csrc fused-op kernels into a native Windows torch.ops._C extension.

This TheRock Windows torch ships ATen/cuda & c10/cuda as non-redirecting CUDA stubs (Linux
ROCm redirects them to HIP), and torch's hipify refuses to rewrite the vLLM sources here. We
hipify them ourselves by applying torch's OWN CUDA_TO_HIP_MAPPINGS with word boundaries
(so e.g. `cub` is not rewritten inside `hipcub`), longest-token-first. Redirect shim headers
are also placed first on the include path as a safety net for any include path we miss.
"""
import os
import re
import shutil
import sys
import time

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp

import torch
from torch.utils import cpp_extension
from torch.utils.hipify.cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS

from torch.utils.hipify import hipify_python as _hp
_orig = _hp.hipify
def _no_none(*a, **k):
    r = _orig(*a, **k)
    try:
        for key, v in r.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception:
        pass
    return r
_hp.hipify = _no_none

VLLM_CSRC = wp.vllm_csrc()
HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "shim")
HIPDIR = wp.build_dir("vw_cext_hip")
BUILD_DIR = wp.build_dir("vw_cext_build", "VLLM_WIN_C_DIR", clean=True)
DEVICE_LIB = wp.device_lib()

# --- redirect shims (include safety net) ---
SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n"
                              "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "ATen/cuda/Exceptions.h": "#include <ATen/hip/Exceptions.h>\n",
    "c10/cuda/CUDAGuard.h": "#include <c10/hip/HIPGuard.h>\n"
                           "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAStream.h": "#include <c10/hip/HIPStream.h>\n"
                            "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
}
for rel, body in SHIMS.items():
    dst = os.path.join(SHIM, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write("#pragma once\n" + body)

# --- hipify using torch's REAL pytorch substitution engine, bypassing its (Windows-broken)
# file-orchestrator. RE_PYTORCH_PREPROCESSOR + PYTORCH_MAP is exactly what hipify applies for
# pytorch extensions: it rewrites cudaStream_t->hipStream_t, __nv_bfloat16->__hip_bfloat16,
# OptionalCUDAGuard->OptionalHIPGuardMasqueradingAsCUDA, getCurrentCUDAStream->...Masquerading,
# while KEEPING the at::cuda:: namespace (correct masquerade). Include redirects are handled by
# the shim headers above. The regex requires \W on both sides, so no partial/mangled matches.
from torch.utils.hipify.hipify_python import RE_PYTORCH_PREPROCESSOR, PYTORCH_MAP

print("torch", torch.__version__, "hip", torch.version.hip)
print(f"=== hipify (PYTORCH_MAP engine) csrc -> {HIPDIR} ===")
shutil.rmtree(HIPDIR, ignore_errors=True)
shutil.copytree(VLLM_CSRC, HIPDIR)

def _pt(mo):
    return str(PYTORCH_MAP[mo.group(1)])

n = 0
for dp, _, fns in os.walk(HIPDIR):
    for fn in fns:
        if not fn.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc")):
            continue
        p = os.path.join(dp, fn)
        s = open(p, encoding="utf-8", errors="ignore").read()
        s2 = RE_PYTORCH_PREPROCESSOR.sub(_pt, s)
        if s2 != s:
            open(p, "w", encoding="utf-8", newline="\n").write(s2)
            n += 1
print("rewrote", n, "files")
_chk = open(os.path.join(HIPDIR, "activation_kernels.cu"), encoding="utf-8", errors="ignore").read()
print("check: hipStream_t =", "hipStream_t" in _chk, "| cudaStream_t left =", "cudaStream_t" in _chk,
      "| MasqueradingAsCUDA =", "MasqueradingAsCUDA" in _chk)

src = [os.path.join(HIPDIR, "activation_kernels.cu"),
       os.path.join(HIPDIR, "layernorm_kernels.cu"),
       os.path.join(HIPDIR, "pos_encoding_kernels.cu"),
       # Native W4A16 GEMM (exllama path). Has a dedicated M<=50 (MAX_Q_GEMM_ROWS) kernel for
       # single-stream decode -- the general fix for the conch fallback being ~21x off at M=1.
       os.path.join(HIPDIR, "quantization", "gptq", "q_gemm.cu"),
       os.path.join(HERE, "win_c_bindings.cu")]
print("=== compiling ===")
sys.stdout.flush()
t0 = time.perf_counter()
cpp_extension.load(
    name="vllm_win_C", sources=src,
    extra_include_paths=[SHIM, HIPDIR],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        # CUDA_VERSION was hipified to TORCH_HIP_VERSION; it gates an NVIDIA Blackwell/CUDA-12.9
        # fast path that must be OFF on RDNA3. Define 0 so `>= 12090` is false.
        "-DTORCH_HIP_VERSION=0",
        # vLLM's real ROCm build defines USE_ROCM for all csrc; q_gemm.cu needs it to take the
        # hipblas/masquerade path (and to skip the NVIDIA tensor-core m8/m16 variants).
        "-DUSE_ROCM=1",
        f"-I{SHIM}", f"-I{HIPDIR}",
    ],
    # hipblas/rocblas are needed by q_gemm.cu's reconstruct path (M>50 prefill).
    extra_ldflags=["/LIBPATH:" + wp.hip_lib(), "hipblas.lib", "rocblas.lib", "amdhip64.lib"],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")

import torch.nn.functional as F
x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
out = torch.empty(64, 2048, device="cuda", dtype=torch.float16)
torch.ops._C.silu_and_mul(out, x)
print("SILU_OK", bool((out - (F.silu(x[..., :2048]) * x[..., 2048:])).abs().max().item() < 1e-2))
w = torch.randn(4096, device="cuda", dtype=torch.float16)
rn = torch.empty_like(x)
torch.ops._C.rms_norm(rn, x, w, 1e-6)
xr = x.float(); refn = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16) * w
print("RMSNORM_OK", bool((rn - refn).abs().max().item() < 2e-2))
print("CEXT_OK")
