# Contributing

Thanks for wanting to help! The most valuable contributions for this project
are **GPU verification reports**, because the stack is only validated on a
small number of cards so far.

## 1. Verify the stack on YOUR GPU (most wanted)

Run `INSTALL.bat`, then `VERIFY.bat`, and open a GPU verification issue with
the results:

1. Your GPU: exact model, driver version, Windows version.
2. Output of `VERIFY.bat` (paste the whole log):
   - step [1/3]: HIP version + GPU name + VRAM
   - step [2/3]: rocBLAS FP16 Gflops
   - step [3/3]: `prompt_tokens`, `generated_tokens`, `overall_tok_s`
3. Which model you ran (default is `cyankiwi/Qwen3.5-4B-AWQ-4bit`).
4. Anything unusual (warnings, retries, extra steps).

Use the [GPU verification template](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/issues/new?template=gpu_verification.yml)
so the data lands in a consistent format.

Cards of special interest (not yet confirmed):

- RX 6800 / 6900 / 6950 (gfx1030) — see [issue #2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/issues/2)
- RX 6400 / 6500 XT (4 GB VRAM — the installer now picks a smaller default model)
- Mobile RDNA2 (RX 6600M/6650M/6700M/6800M/6850M XT)
- RDNA3 (RX 7600–7900) and RDNA4 (RX 9000): **experimental** — see `docs/MULTIARCH.md`
  before trying; the release archives for those families are not published yet.

## 2. Report a bug

- Check the [troubleshooting table](README.md#troubleshooting) first.
- Open a bug report with the full transcript (`C:\_release_staging\setup_transcript.log`
  is written by the installer) and the exact command you ran.

## 3. Contribute code

- Small, focused PRs against `main`.
- Keep the one-click philosophy: every change to `INSTALL.ps1` must stay
  idempotent and must not add steps a user has to think about.
- The PowerShell scripts are linted by CI (PSScriptAnalyzer, severity Error)
  and `MANIFEST.json` is validated against the GitHub release API by
  `.github/workflows/validate.yml` — run the same checks locally:
  ```powershell
  Invoke-ScriptAnalyzer -Path INSTALL.ps1,UNINSTALL.ps1,scripts\make_checksums.ps1 -Severity Error
  ```
- `config.bat` is generated and git-ignored — never commit it, never rely on
  machine-specific paths in the repo (see `scripts/serve.py` for the pattern:
  error out clearly when config is missing).

## 4. Rebuild the native kernels

`kernels\build_kernels.ps1` recompiles `kernels/src/gemv_w4.cu` per GPU family
(needs HIP SDK + MSVC). The shipped fat binaries cover all RDNA2; only rebuild
for non-RDNA2 targets or exotic torch ABIs.

## Code of conduct

Be respectful and constructive. No affiliation with AMD; this is a community
effort built on ROCm/TheRock, PyTorch ROCm and vLLM.