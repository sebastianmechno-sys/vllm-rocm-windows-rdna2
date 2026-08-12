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

## License

This project is licensed under the Apache 2.0 License.