# Security Policy

## Reporting a vulnerability

If you find a security issue in this project (installer scripts, plugin code,
or the packaging pipeline), **do not open a public issue**. Report it privately
so it can be fixed before disclosure:

- Open a [private advisory](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/security/advisories/new), or
- Email the maintainer through the contact info on the GitHub profile.

Please include: affected version, what the issue is, how to reproduce it, and
(if known) a suggested fix. You will get an acknowledgement within 7 days.

## Supply-chain notes (important)

This stack downloads and runs **prebuilt binaries** (~6 GB) published as GitHub
release assets:

- The binaries are **not code-signed** and there is no upstream trust anchor
  (AMD does not ship ROCm for these cards on Windows).
- The installer verifies **every downloaded part against `SHA256SUMS.txt`**
  from the same release, so a corrupted or tampered-with download is rejected
  before it is extracted. This protects you from broken downloads and
  man-in-the-middle substitution **of release assets**, but it cannot protect
  you if the release itself (and its checksums) is malicious.
- Before trusting a release, verify the `SHA256SUMS.txt` file itself was
  published by the maintainer (it lives on the official Releases tab) and, for
  high-assurance setups, re-check the hashes of the extracted files you care
  about with `Get-FileHash`.

### What you can do to raise trust

- Review the build inputs: the source for the native kernels lives in
  `kernels/` and the plugin in `plugin_overrides/` (plus the upstream
  `vllm-rocm-windows` code the stack is packaged from).
- Rebuild the native kernels yourself with `kernels\build_kernels.ps1` if you
  have the HIP SDK — the prebuilt `.pyd` is only a convenience.
- Run the stack on an isolated machine / VM with no personal data if you are
  concerned.

## Supported versions

| Version | Supported |
|---|---|
| main (latest) | ✅ |
| V2.1 | ✅ security fixes |
| V2.0 | ✅ security fixes |
| v1.1.0 and older | ❌ |

## Scope

Security-sensitive surface: `INSTALL.ps1` (downloads + extraction + elevation),
`UNINSTALL.ps1` (elevated deletion), `make_checksums.ps1`, the GitHub Actions
workflows, and any code that executes downloaded binaries (`serve.py`,
`benchmark.py`, the native kernel `.pyd` files).