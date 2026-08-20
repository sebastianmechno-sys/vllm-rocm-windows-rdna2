# =============================================================================
# validate_release.ps1 - checks MANIFEST.json against the GitHub release API.
#
# Verifies that:
#   - MANIFEST.json parses and has the expected variant structure
#   - every asset part of the DEFAULT variant (rdna2) exists on the release
#     of the default tag (the installer must not hit 404s)
#   - SHA256SUMS.txt is published next to the archives
#   - rdna3/rdna4 variants are reported as warnings when their assets are not
#     published yet (expected until hardware validation happens)
#
# Usage:
#   scripts\validate_release.ps1                     # default repo/tag
#   scripts\validate_release.ps1 -Tag V2.1 -Owner you -Repo your-fork
# =============================================================================
param(
    [string]$Owner = "sebastianmechno-sys",
    [string]$Repo = "vllm-rocm-windows-rdna2",
    [string]$Tag = "V2.0"
)
$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path (Split-Path -Parent $here) "MANIFEST.json"
if (-not (Test-Path $manifestPath)) { Write-Host "ERROR: MANIFEST.json not found" -ForegroundColor Red; exit 1 }

$cfg = Get-Content $manifestPath -Raw | ConvertFrom-Json
if (-not $cfg.variants) { Write-Host "ERROR: MANIFEST.json has no 'variants'" -ForegroundColor Red; exit 1 }

$headers = @{ "User-Agent" = "vllm-rocm-windows-rdna2-validate" }
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/releases/tags/$Tag" -Headers $headers
$have = @{}
$release.assets | ForEach-Object { $have[$_.name] = $true }
Write-Host "release '$Tag': $($release.assets.Count) assets"

$fail = 0
$warn = 0
foreach ($vname in $cfg.variants.PSObject.Properties.Name) {
    $v = $cfg.variants.$vname
    Write-Host "`nvariant $vname (override $($v.override)):"
    foreach ($a in $v.assets) {
        foreach ($f in $a.files) {
            if ($have.ContainsKey($f)) {
                Write-Host "  OK   $f" -ForegroundColor Green
            } elseif ($vname -eq "rdna2") {
                Write-Host "  MISS $f" -ForegroundColor Red
                $fail++
            } else {
                Write-Host "  warn $f (experimental family - expected until published)" -ForegroundColor Yellow
                $warn++
            }
        }
    }
}

if (-not $have.ContainsKey("SHA256SUMS.txt")) {
    Write-Host "`nMISS SHA256SUMS.txt - integrity check will be disabled for this release!" -ForegroundColor Red
    $fail++
} else {
    Write-Host "`nOK   SHA256SUMS.txt" -ForegroundColor Green
}

Write-Host "`nsummary: $fail problem(s), $warn experimental warning(s)"
if ($fail -gt 0) { exit 1 }
exit 0