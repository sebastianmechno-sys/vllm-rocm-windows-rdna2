# vllm-rocm-windows-rdna2-oneclick

[![Windows](https://img.shields.io/badge/OS-Windows%2010%2F11-0078D6?style=for-the-badge&logo=windows)]()
[![ROCm](https://img.shields.io/badge/ROCm-7.15%20TheRock-FF0000?style=for-the-badge&logo=amd)]()
[![RDNA2](https://img.shields.io/badge/GPU-RDNA2%20RX%206400%E2%80%936950-000000?style=for-the-badge)]()
[![vLLM](https://img.shields.io/badge/vLLM-0.19.1-00C853?style=for-the-badge)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2Brocm7.15-EE4C2C?style=for-the-badge&logo=pytorch)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)

Native vLLM + ROCm 7.15 (TheRock) for the whole AMD Radeon RDNA2 family on
Windows — **no WSL2, no NVIDIA, no compiler**. One-click installer, everything
prebuilt, and an OpenAI-compatible chat server that looks and works like the
NVIDIA stack.

**Verified on AMD Radeon RX 6750 XT 12 GB (gfx1031) — Windows 11 native — August 2026**

| Result | Number |
|---|---|
| rocBLAS FP16 GEMM (native bench) | **25 674 Gflops ≈ 26 TFLOPS** |
| vLLM decode, Qwen3.5-4B 4-bit | **~58-62 tok/s** (8.3 → 62.5 = 7.5× optimized) |

---

## Verification — real output from the RX 6750 XT

### 1. ROCm detects the GPU (native Windows process, no WSL)

![gpu detection](assets/01_gpu_detection.png)

### 2. Raw GPU power — native rocblas-bench.exe, FP16 GEMM 4096³

![rocblas bench](assets/02_rocblas_bench.png)

### 3. vLLM tuned decode

![vllm tps](assets/03_vllm_tps.png)

---

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 (Windows 11 recommended; `tar` must support zstd — automatic on Win11) |
| GPU | **AMD RDNA2** — RX 6400 / 6500 / 6600 / 6650 / 6700 / 6750 / 6800 / 6900 / 6950 (all XT/M variants), 8+ GB VRAM for the 4B model |
| Driver | AMD Software: Adrenalin Edition (the normal gaming driver) |
| Disk | ~25 GB free on `C:` |
| Internet | only during install (~6 GB: stack ~2.3 GB + model ~3.8 GB) |
| Admin | one UAC click (installer auto-elevates) |

No compiler, no ROCm installer, no manual setup — everything ships prebuilt.

## Quick start (one-click)

1. Download the repository (ZIP or `git clone`).
2. **Double-click `INSTALL.bat`** — by default it downloads from this repo's
   releases; pass a GitHub username to use your own fork instead. It checks
   GPU + disk, then installs everything it does not already have (re-run is
   always safe and fast):

   | Step | Action |
   |---|---|
   | 1/6 | GPU detection (warns if not RDNA2) + Python 3.11.9 |
   | 2/6 | 4 archives from GitHub Releases → `C:\Python311`, `C:\TheRock`, `C:\vw_*_build` |
   | 3/6 | venv fix + torch self-check |
   | 4/6 | Qwen3.5-4B 4-bit model (skipped if already in HuggingFace cache) |
   | 5/6 | writes `config.bat` (the file to edit to change model later) |
   | 6/6 | verification benchmark |

3. **`CHAT.bat`** → opens the web chat in your browser (starts the model server
   automatically the first time). While the model thinks you see a small
   *Thinking…* spinner; when it's done, only the final answer streams in — the
   internal reasoning stays hidden. Tokens stream live with a tok/s counter,
   everything local on your AMD GPU.
4. **`SERVE.bat`** → starts the model server on its own (OpenAI-compatible API
   on `http://127.0.0.1:8000/v1`, like `vllm serve` on NVIDIA). Use it with any
   OpenAI client, or just run `CHAT.bat`.
5. **`VERIFY.bat`** → all 3 verification checks in one run: ROCm GPU detection,
   native rocBLAS FP16 power (**~26 TFLOPS**) and the full 512-token vLLM
   benchmark (**~58-62 tok/s**).

## Benchmarks

| Test | Config | Result |
|---|---|---|
| rocBLAS FP16 GEMM | 4096×4096×4096, rocblas-bench.exe | **25 674 Gflops (≈26 TFLOPS)** |
| vLLM Qwen3.5-4B decode | 512 tok, greedy, CUDA graphs | **59.4 tok/s** (up to 62.5) |
| Optimization progression | eager fp16 baseline | 8.3 → 62.5 tok/s (**7.5×**) |

![progression](results/benchmark.png)

Full optimization history:

| # | configuration | tok/s |
|---|---|---:|
| 1 | fp16 eager (baseline) | 8.3 |
| 2 | + CUDA graphs + skinny GEMV | 24.4 |
| 3 | + AWQ 4-bit quantization | 29.9 |
| 4 | + native HIP W4 GEMV kernel | 35.9 |
| 5 | + M=1 GEMV for lm_head | 58.1 |
| 6 | + direct-store kernel path | 59.1 |
| 7 | + weight-cast caching | **62.5** |

## How it works

1. **TheRock** builds ROCm (HIP runtime, rocBLAS, Tensile) as native Windows
   binaries — this is what makes ROCm exist on Windows at all.
2. `HSA_OVERRIDE_GFX_VERSION=10.3.0` presents any RDNA2 GPU as gfx1030; the
   HIP kernel ships as a **fat binary (gfx1030 + gfx1031 + gfx1032)** so the
   whole RX 6000 series runs native code.
3. PyTorch 2.12 `+rocm7.15` links against that runtime →
   `torch.cuda.is_available() == True` on RDNA2 Windows.
4. vLLM plugin `vllm_windows_rocm` registers the tuned kernels: native HIP W4
   GEMV for quantized linears, M=1 skinny GEMV for dense ones (including the
   huge tied lm_head), CUDA-graph safe (registered as real torch ops).
5. `INSTALL.bat` (engine: `INSTALL.ps1`, manifest: `MANIFEST.json`) downloads
   the 4 prebuilt archives from GitHub Releases and the model from HuggingFace,
   installs base Python, fixes the venv, verifies with a benchmark. Idempotent:
   it only downloads what is missing.

## Installed layout

```
C:\Python311                                 Python 3.11.9
C:\TheRock\.venv                             torch 2.12+rocm7.15 venv (vLLM 0.19.1)
C:\TheRock\build\dist\rocm                   ROCm runtime libraries
C:\TheRock\ROCM_VLLM_RUNTIME                 vLLM + plugin + rocBLAS + rocblas-bench
C:\vw_cext_build, C:\vw_hipgemv_build        native HIP kernels
%USERPROFILE%\.cache\huggingface             model weights
```

Release archives (this repo's **Releases** tab, tag `v2.0`):

| Archive | Size | Content |
|---|---:|---|
| `the-rock-venv.tar.zst` | 1.34 GB | torch ROCm venv |
| `therock-rocm-dist.tar.zst` | 0.85 GB | ROCm runtime |
| `vllm-stack.tar.zst` | 0.14 GB | vLLM + plugin + rocBLAS + rocblas-bench.exe |
| `native-kernels.tar.zst` | ~1 MB | HIP GEMV kernels (fat binary) |

## Repository structure

```
vllm-rocm-windows-rdna2-oneclick/
├── INSTALL.bat               one-click installer (entry point)
├── INSTALL.ps1               installer engine (downloads, extracts, verifies)
├── CHAT.bat                  opens the web chat (auto-starts the server)
├── SERVE.bat                 starts the model server alone (OpenAI API)
├── VERIFY.bat                all 3 verification checks in one run
├── chat.html                 the browser chat UI (Thinking spinner + tok/s)
├── MANIFEST.json             release archive names + extract targets
├── plugin_overrides/         tuned plugin modules (awq_gemv, bf16_gemv)
├── kernels/                  native HIP W4 GEMV source + prebuilt fat binary
├── scripts/                  model server entry + benchmark + rebuild helper
├── assets/                   verification screenshots
└── results/                  raw benchmark logs + progression chart
```

## Using a different model

Edit `config.bat` (written by the installer): set `SERVED_MODEL` to the model
folder and `MODEL_NAME` to the name shown in the chat / used by the API, then
run `CHAT.bat` again. The benchmark uses `BENCH_MODEL` (same file):

```bat
C:\TheRock\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('<owner>/<repo>'))"
```

prints the snapshot folder to put in `config.bat`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'vllm._C'` warnings | expected — the Windows plugin loads native kernels instead |
| Python installer exit 1601 | automatic NuGet fallback kicks in; nothing to do |
| extraction "Can't unlink" errors | close stray Python processes, re-run `INSTALL.bat` (it resumes) |
| low tok/s | close other GPU workloads; verify `overall_tok_s` ≥ 55 on a cold GPU |
| non-RDNA2 GPU | installer warns; unvalidated outside RDNA2 |
| `tar` says "Unrecognized archive format" on the `.zst` files | old Windows 10 without zstd — install Windows updates (bsdtar 3.5+ bundles libzstd; Windows 11 is fine) |
| chat page says "Server error" | the model is still loading — wait for "Application startup complete" in the SERVE window (~1 minute the first time) |
| custom debugging | `set BENCH_MODE=eager`, `set VLLM_WIN_HIPGEMV=0`, `set VLLM_WIN_BF16_GEMV=0` |

## Rebuild the kernel

`scripts\rebuild_kernel.py` recompiles `kernels/src/gemv_w4.cu` (needs HIP SDK
+ MSVC). Only for non-RDNA2 targets or exotic torch ABIs; the shipped fat
binary covers all RDNA2.

## Acknowledgments

Built on ROCm/TheRock, PyTorch ROCm and the vLLM project. Not affiliated with AMD.

## License

[Apache License 2.0](LICENSE)
