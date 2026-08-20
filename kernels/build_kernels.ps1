# =============================================================================
# build_kernels.ps1 - rebuild the native HIP W4 GEMV kernel per GPU family.
#
# Wraps scripts\rebuild_kernel.py with the per-family gfx target set and
# output directory, so the packaged fat binary can be regenerated for any
# family (RDNA2 is the validated default; RDNA3/RDNA4 are experimental).
#
# Usage:
#   kernels\build_kernels.ps1                # rdna2 -> C:\vw_hipgemv_build
#   kernels\build_kernels.ps1 -Family rdna3  # gfx1100+1101+1102 -> C:\vw_hipgemv_build_rdna3
#   kernels\build_kernels.ps1 -Family rdna4  # gfx1200+1201 -> C:\vw_hipgemv_build_rdna4
#
# Requires: HIP SDK (C:\HIP-SDK), MSVC Build Tools, and the ROCm venv python
# (C:\TheRock\.venv by default).
# =============================================================================
param(
    [ValidateSet("rdna2","rdna3","rdna4")]
    [string]$Family = "rdna2",
    [string]$VenvPython = "C:\TheRock\.venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'

$archs = @{
    rdna2 = "gfx1030,gfx1031,gfx1032"
    rdna3 = "gfx1100,gfx1101,gfx1102"
    rdna4 = "gfx1200,gfx1201"
}
$dirs = @{
    rdna2 = "C:\vw_hipgemv_build"
    rdna3 = "C:\vw_hipgemv_build_rdna3"
    rdna4 = "C:\vw_hipgemv_build_rdna4"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at $VenvPython - run INSTALL.bat first." -ForegroundColor Red
    exit 1
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir = $dirs[$Family]
New-Item -ItemType Directory -Force $outDir | Out-Null

Write-Host "[build] family $Family archs=$($archs[$Family])" -ForegroundColor Cyan
Write-Host "[build] output $outDir" -ForegroundColor Cyan
$env:KERNEL_ARCHS = $archs[$Family]
$env:TORCH_EXTENSIONS_DIR = $outDir
$env:VLLM_WIN_HIPGEMV_DIR = Join-Path $outDir "gemv_w4_hip"

& $VenvPython "$here\..\scripts\rebuild_kernel.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[build] FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "[build] OK - kernels for $Family in $outDir\gemv_w4_hip" -ForegroundColor Green
$overrides = @{ rdna2 = "10.3.1"; rdna3 = "11.0.0"; rdna4 = "12.0.0" }
Write-Host "[build] remember: set HSA_OVERRIDE_GFX_VERSION=$($overrides[$Family]) for this family (INSTALL.bat does it automatically)" -ForegroundColor Yellow