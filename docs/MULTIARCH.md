# Multi-architecture support (RDNA2 / RDNA3 / RDNA4)

The one-click stack is organized around **GPU families**. `MANIFEST.json`
defines one asset set per family; `INSTALL.ps1` auto-detects the family from
the GPU name (or takes `-Variant`), downloads the matching archives and writes
the right `HSA_OVERRIDE_GFX_VERSION` into `config.bat`.

| Family | GPUs | gfx targets | Override | Status |
|---|---|---|---|---|
| **rdna2** | RX 6400–6950 (desktop + M), Radeon Pro V620 | gfx1030, gfx1031, gfx1032 | `10.3.1` | ✅ validated (RX 6750 XT); others pending community verification |
| **rdna3** | RX 7600–7900 (desktop + M) | gfx1100, gfx1101, gfx1102 | `11.0.0` | ⚠️ experimental — code + build pipeline ready, archives NOT published, needs hardware validation |
| **rdna4** | RX 9000 | gfx1200, gfx1201 | `12.0.0` | ⚠️ experimental — feasibility unconfirmed, needs hardware validation |

## How a family is packaged

1. **ROCm runtime** (`therock-rocm-dist`): TheRock build with the family's
   Tensile/rocBLAS libraries (the shipped rdna2 dist is gfx1031-only, which is
   why every RDNA2 card is presented as gfx1031 via the override).
2. **torch venv** (`the-rock-venv`): PyTorch 2.12 `+rocm7.15` built against
   that runtime.
3. **Native kernels** (`native-kernels`): the HIP GEMV `.pyd` compiled as a
   **fat binary** for the family's gfx targets (`kernels\build_kernels.ps1`).
4. **Installer**: per-family `HSA_OVERRIDE_GFX_VERSION` + `VLLM_WIN_HIPGEMV_DIR`
   written to `config.bat`.

## Building a family's kernels

```powershell
powershell -ExecutionPolicy Bypass -File kernels\build_kernels.ps1 -Family rdna3
```

Requires HIP SDK + MSVC (only for non-RDNA2 families or exotic torch ABIs;
the shipped rdna2 fat binary covers all RDNA2).

## Releasing a family

Follow `docs/RELEASING.md`. The archive names must match the variant block in
`MANIFEST.json` (`rdna3-*`, `rdna4-*` prefixes), and `SHA256SUMS.txt` must be
uploaded to the same release. `scripts\validate_release.ps1` treats missing
experimental-family assets as warnings, not errors.

## Validation checklist for RDNA3 / RDNA4 (when you have the hardware)

- [ ] `VERIFY.bat` step [1/3]: HIP detects the GPU (no WSL), correct name/VRAM
- [ ] step [2/3]: rocBLAS FP16 GEMM completes (no Tensile fallback errors)
- [ ] step [3/3]: `overall_tok_s` sane (≥ 40 on a cold GPU with the 4B model)
- [ ] `CHAT.bat` streams a full answer with thinking spinner + tok/s counter
- [ ] CPU/GPU temps stable, no driver resets (TDR) during a 512-token run
- [ ] Report results via the GPU verification issue template

Until those boxes are ticked, the family stays **experimental** and its
archives are not published.

## Troubleshooting per family

| Symptom | Cause / fix |
|---|---|
| `download failed: ...rdna3-...` | archives not published yet — family still experimental |
| `checksum mismatch` | corrupted download; re-run `INSTALL.bat` (parts are re-fetched) |
| torch self-check fails on RDNA3/4 | wrong venv/arch pair — the venv must match the family (do not reuse the rdna2 venv) |
| driver timeout (TDR) | known on RDNA2 with some drivers; update Adrenalin, lower `--max-model-len` |