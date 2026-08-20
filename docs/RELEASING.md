# Releasing a new version

This project ships prebuilt binaries as GitHub release assets. The installer
(`INSTALL.ps1`) fetches them by tag and verifies them against
`SHA256SUMS.txt` on the same release. The default tag lives in `INSTALL.ps1`
(`$Tag`).

## When a new tag is needed

- Any change to the packaged stack (venv, ROCm dist, vLLM stack, native
  kernels) requires a new tag + full re-upload of that family's archives.
- Changes to the installer/plugin code **in the repo** do NOT require new
  archives — the user downloads the repo itself fresh.

## 1. Build the archives

Archives are built from the working machine (see `C:\_release_staging`):

| Archive | Content |
|---|---|
| `the-rock-venv.tar.zst` | `C:\TheRock\.venv` (torch ROCm venv) |
| `therock-rocm-dist.tar.zst` | `C:\TheRock\build\dist\rocm` |
| `vllm-stack.tar.zst` | `C:\TheRock\ROCM_VLLM_RUNTIME` |
| `native-kernels.tar.zst` | `C:\vw_cext_build` + `C:\vw_hipgemv_build` |

```powershell
tar --zstd -cf the-rock-venv.tar.zst -C C:\TheRock .venv
tar --zstd -cf therock-rocm-dist.tar.zst -C C:\TheRock build\dist\rocm
tar --zstd -cf vllm-stack.tar.zst -C C:\TheRock ROCM_VLLM_RUNTIME
tar --zstd -cf native-kernels.tar.zst -C C:\ vw_cext_build vw_hipgemv_build
```

Split anything > ~700 MB into parts (GitHub upload limit), matching the
`files` arrays in `MANIFEST.json`:

```powershell
# 2 parts of 700 MB (byte-identical rejoin by the installer)
& "C:\Program Files\7-Zip\7z.exe" a -tzip -v700m ... # or use the split tool of choice
# simplest: use 7-Zip / gsplit producing the exact .part-01/.part-02 names
```

> Naming rule: `name.tar.zst.part-XX`. The installer rejoins parts into
> `name.tar.zst` and extracts.

## 2. Per-GPU-family archives (RDNA3 / RDNA4)

For a new family (`rdna3` gfx1100, `rdna4` gfx1200), the archive names must
match the variant block in `MANIFEST.json` (`rdna3-*`, `rdna4-*` prefixes),
and the kernels must be built for that family's gfx targets:

```powershell
powershell -ExecutionPolicy Bypass -File kernels\build_kernels.ps1 -Family rdna3
```

See `docs/MULTIARCH.md` — these families are experimental and MUST NOT be
released before hardware validation.

## 3. Generate and upload SHA256SUMS.txt

```powershell
# from the repo root, with the archives (or parts) in C:\_release_staging
powershell -ExecutionPolicy Bypass -File scripts\make_checksums.ps1
```

Then upload **`SHA256SUMS.txt`** as an asset of the SAME release that hosts
the archives (it lives next to them, never alone). Also possible:

```powershell
scripts\make_checksums.ps1 -DownloadMissing   # fetches missing parts from the release
```

## 4. Create the GitHub release

```bash
gh release create V2.1 --title "v2.1" --notes "..." \
  SHA256SUMS.txt \
  the-rock-venv.tar.zst.part-01 the-rock-venv.tar.zst.part-02 \
  therock-rocm-dist.tar.zst.part-01 therock-rocm-dist.tar.zst.part-02 \
  vllm-stack.tar.zst native-kernels.tar.zst
```

(or use the web UI: Releases → Draft a new release → attach binaries)

## 5. Update the default tag

If the new tag becomes the default, update `$Tag` in `INSTALL.ps1` and the
`-Tag` default of `scripts\make_checksums.ps1`, then commit + push.

## 6. Validate

`.github/workflows/validate.yml` checks (on push/PR) that every asset in
`MANIFEST.json` exists on the default-tag release and that `SHA256SUMS.txt`
is present. Run it locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate_release.ps1 -Tag V2.1
```