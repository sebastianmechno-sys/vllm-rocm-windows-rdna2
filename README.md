# RDNA2-vLLM-ROCm-Core (Windows 11)

[![OS - Windows 11](https://img.shields.io/badge/OS-Windows_11-blue?style=flat&logo=windows11)](https://www.microsoft.com/windows)
[![AMD ROCm - 7.15](https://img.shields.io/badge/ROCm-7.15_(TheRock)-red?style=flat&logo=amd)](https://www.amd.com)
[![PyTorch - 2.12.0](https://img.shields.io/badge/PyTorch-2.12.0%2Brocm7.15-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org)
[![vLLM - 0.19.1](https://img.shields.io/badge/vLLM-0.19.1-orange)](https://github.com/vllm-project/vllm)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

Native **vLLM** and **ROCm 7.x** runtime setup for **AMD Radeon RX 6000 Series (RDNA2)** GPUs on Windows 11 — No WSL2 or Linux required!

Tested and verified on an **AMD Radeon RX 6750 XT (12GB VRAM / `gfx1031`)**.

---

## Technical Stack

- **OS:** Windows 11 Native
- **GPU:** AMD Radeon RX 6000 Series / RX 6750 XT (`gfx1031`)
- **ROCm Core:** ROCm 7.15 (TheRock runtime) with `rocBLAS` binaries built for `gfx1031`
- **PyTorch:** `2.12.0+rocm7.15`
- **vLLM Engine:** `v0.19.1`

---

## Quick Start

1. Extract the release archive.
2. Run `setup_prima_volta.bat` once to configure environment variables.
3. Run `avvia.bat` (or execute `python run_first2.py`) to launch the engine.

---

## Benchmarks & Verification

- **`rocblas-bench` FP16:** ~25.8 TFLOPS (5.32 ms)
- **`rocblas-bench` FP32:** ~12.7 TFLOPS (10.8 ms)
- **vLLM Inference:** `FIRST_TOKEN_OK` verified on `facebook/opt-125m` (~24.6 tok/s)

---

Known Issues — Experimental Status
This is an experimental reference implementation. It is expected to have bugs on other hardware.

Current status: Verified and working on RX 6750 XT 12GB (gfx1031) — FIRST_TOKEN_OK, 25.9 TFLOPS FP16, 54.20 tok/s.

Known limitations:

Other RDNA2 cards (RX 6600, 6600 XT, 6700 XT) are not yet tested — may need different HSA_OVERRIDE_GFX_VERSION (10.3.0 vs 10.3.1) or rocBLAS rebuild for gfx1030.
Windows ZMQ IPC bug fixed in v1.1.0 by forcing VLLM_ENABLE_V1_MULTIPROCESSING=0 to use TCP. Other Windows IPC limitations may appear on different models.
enforce_eager=True is required — torch.compile and CUDA graphs are disabled on RDNA2 Windows.
TheRock 90GB full build is not included in release (only 708MB minimal runtime). Developers need to build TheRock locally for full rebuild — see docs/BUILD_ROCBLAS.md.
FP8 and AWQ quantization not tested on RDNA2 Windows yet.
Multi-GPU (HIP_VISIBLE_DEVICES) not tested.
If it boots on your RDNA2 card, please open an Issue with your GPU model, gfx version and logs. Contributions welcome — this is a reference for AMD engineers to re-enable RDNA2 in official ROCm 7 Windows builds.

---

## License

This project is licensed under the Apache 2.0 License.