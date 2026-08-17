"""Rebuild the native HIP W4 GEMV kernel from source.

Use this if the prebuilt pyd does not work on your machine (different gfx target,
torch ABI, etc.). Requires: HIP SDK (C:\HIP-SDK or Program Files\AMD\ROCm), MSVC
Build Tools, and the ROCm venv python on the interpreter you rebuild with.

Output directory: C:\vw_hipgemv_build\gemv_w4_hip  (the path the plugin loads).
"""
import os
import sys

os.environ.setdefault("TORCH_EXTENSIONS_DIR", r"C:\vw_hipgemv_build")

import torch  # noqa: E402
from torch.utils import cpp_extension  # noqa: E402
from torch.utils.hipify import hipify_python as _hp  # noqa: E402

# hipify on Windows returns hipified_path=None for extra_files -> guard it.
_orig_hipify = _hp.hipify


def _no_none(*a, **k):
    r = _orig_hipify(*a, **k)
    try:
        for key, v in r.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception:
        pass
    return r


_hp.hipify = _no_none

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "kernels", "src", "gemv_w4.cu")

# device-libs bitcode (comes with the HIP SDK)
HIP_PATH = os.environ.get("HIP_PATH", r"C:\HIP-SDK")
DEVICE_LIB = os.path.join(HIP_PATH, "amdgcn", "bitcode")

mod = cpp_extension.load(
    name="gemv_w4_hip",
    sources=[os.path.abspath(SRC)],
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        "-DUSE_ROCM=1", "-DTORCH_HIP_VERSION=0", "-O3",
    ],
    verbose=True,
)
print("KERNEL_BUILD_OK ->", mod.__file__)
