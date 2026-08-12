"""Insert the windows_rocm_rocm bootstrap import at the top of vLLM's package __init__,
so the Windows/ROCm compatibility shims are installed before any vLLM submodule loads
torch.distributed. Idempotent.

Usage: python tools/patch_vllm.py [path-to-vllm-checkout]   (default: ./vllm)
"""
import sys
import os

MARK = "vllm_windows_rocm.bootstrap"
BLOCK = (
    "\n# --- vLLM-on-Windows-ROCm: install the single-process torch.distributed shim and\n"
    "# _C op fallbacks before any vllm submodule that imports torch.distributed is loaded.\n"
    "try:\n"
    "    import vllm_windows_rocm.bootstrap  # noqa: F401\n"
    "except Exception:\n"
    "    pass\n"
)


def main(argv):
    vllm_root = argv[0] if argv else "vllm"
    init_py = os.path.join(vllm_root, "vllm", "__init__.py")
    if not os.path.isfile(init_py):
        print(f"not found: {init_py}", file=sys.stderr)
        return 1
    src = open(init_py, encoding="utf-8").read()
    if MARK in src:
        print("already patched")
        return 0
    anchor = "from .version import __version__"
    idx = src.find(anchor)
    if idx == -1:
        # fall back: prepend
        new = BLOCK + src
    else:
        eol = src.find("\n", idx)
        new = src[: eol + 1] + BLOCK + src[eol + 1 :]
    open(init_py, "w", encoding="utf-8", newline="\n").write(new)
    print("patched", init_py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
