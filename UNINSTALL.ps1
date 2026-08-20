# =============================================================================
# UNINSTALL.ps1 - clean removal of the installed stack.
#
# Removes (in order):
#   C:\TheRock                              runtime + venv + model server
#   C:\vw_cext_build, C:\vw_hipgemv_build   native HIP kernels
#   config.bat (next to this script)        generated model configuration
#   optionally: C:\Python311 and the HuggingFace model cache (prompted)
#
# The project folder itself (repo download) is kept - delete it manually if
# you no longer need it. No registry entries are written, so nothing else
# needs cleaning.
# =============================================================================
$ErrorActionPreference = 'Stop'

# -------------------------------------------------------- self-elevate -------
$isAdmin = (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[uninstall] requesting Administrator rights (UAC prompt)..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",("`"{0}`"" -f $MyInvocation.MyCommand.Path)
    exit
}

function Log($m) { Write-Host "[uninstall] $m" -ForegroundColor Cyan }

Log "close any open SERVE / CHAT window (model server) before continuing."
Read-Host "Press Enter to continue, or Ctrl+C to abort"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$targets = @(
    "C:\TheRock",
    "C:\vw_cext_build",
    "C:\vw_hipgemv_build",
    "C:\vw_hipgemv_build_rdna3",
    "C:\vw_hipgemv_build_rdna4"
)
foreach ($t in $targets) {
    if (Test-Path $t) {
        try {
            Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction Stop
            Log "removed $t"
        } catch {
            Write-Host "[uninstall] WARNING: could not remove $t (still in use?) - $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        Log "skip (not present): $t"
    }
}

$cfg = Join-Path $here "config.bat"
if (Test-Path $cfg) { Remove-Item -LiteralPath $cfg -Force; Log "removed $cfg" } else { Log "skip (not present): config.bat" }

if (Test-Path "C:\Python311") {
    $r = Read-Host "Also remove base Python (C:\Python311)? [y/N]"
    if ($r -match '^[yY]') {
        Remove-Item -LiteralPath "C:\Python311" -Recurse -Force -ErrorAction SilentlyContinue
        Log "removed C:\Python311"
    } else {
        Log "kept C:\Python311 (you can uninstall it from Windows Settings if you want)"
    }
}

$hfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
if (Test-Path $hfCache) {
    $r = Read-Host "Also remove the HuggingFace model cache ($hfCache)? [y/N]"
    if ($r -match '^[yY]') {
        Remove-Item -LiteralPath $hfCache -Recurse -Force -ErrorAction SilentlyContinue
        Log "removed HuggingFace cache"
    } else {
        Log "kept HuggingFace cache"
    }
}

Log "UNINSTALL COMPLETE - the stack has been removed."