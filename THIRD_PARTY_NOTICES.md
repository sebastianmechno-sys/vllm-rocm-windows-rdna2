# Third-Party Notices

This project includes and redistributes software from the following third
parties. All licenses are permissive; this file exists to comply with their
redistribution terms (keep notices, state changes, provide license copies).

## Code included in this repository

### MrPie (ThePie88) — W4 GEMV / AWQ / bf16 kernels
- Files: `plugin_overrides/awq_gemv.py`, `plugin_overrides/bf16_gemv.py`, `kernels/src/gemv_w4.cu`
- Source: https://github.com/ThePie88/vLLM-ROCm-Windows
- License: Apache License 2.0 (Copyright 2026 ThePie88)
- Changes: adapted for RDNA2 (gfx1030/1031/1032) fat-binary build, Windows
  paths, torch op registration and RDNA2 tuning by sebastianmechno-sys.
  A copy of the Apache License 2.0 is in `LICENSE` (same text applies).

## Prebuilt binaries shipped in the GitHub Release archives

The release archives (`the-rock-venv`, `therock-rocm-dist`, `vllm-stack`,
`native-kernels`) contain unmodified-or-build-output binaries from:

| Component | Upstream | License |
|---|---|---|
| vLLM | https://github.com/vllm-project/vllm | Apache 2.0 |
| PyTorch (+rocm) | https://github.com/pytorch/pytorch | BSD-3-Clause |
| ROCm / TheRock (HIP runtime, rocBLAS, Tensile, hipBLASLt, comgr, amdhip64) | https://github.com/ROCm/TheRock | Apache 2.0 / MIT (per component) |
| Python 3.11.9 | https://www.python.org | PSF License 2.0 |
| Python packages in the venv | PyPI (pip, transformers, fastapi, uvicorn, triton, ...) | per-package (BSD / MIT / Apache 2.0) |

Upstream license texts are preserved inside the archives themselves (e.g.
`*.dist-info/LICENSE`, `*.dist-info/LICENSE.txt`, ROCm copyright headers).
Full per-package license dump can be produced with:
`C:\TheRock\.venv\Scripts\python.exe -m piplicenses --format=markdown`
(package `pip-licenses`, not bundled by default).

## Downloaded at install time (NOT redistributed by this project)

- LLM weights (e.g. `cyankiwi/Qwen3.5-4B-AWQ-4bit`, `Qwen/Qwen2.5-1.5B-Instruct-AWQ`)
  are downloaded directly from Hugging Face by the installer and remain under
  their own model licenses (Qwen models: Alibaba Qwen license / Apache 2.0
  depending on the model). This repository ships no model weights.

## Fonts and assets

- `chat.html` loads Inter and Space Grotesk from Google Fonts at runtime
  (SIL Open Font License 1.1). No font files are redistributed.
- Badges in `README.md` are rendered by shields.io at view time.

## Trademarks

AMD, Radeon, ROCm and Instinct are trademarks of Advanced Micro Devices, Inc.
vLLM is a project of the vLLM community (UC Berkeley Sky Computing Lab).
Qwen is a trademark of Alibaba Cloud. This project is an independent
community work: it is **not affiliated with, endorsed by, or sponsored by**
AMD, the vLLM project, Alibaba or Google. All names are used only to
describe compatibility and origin (nominative fair use).
