# =============================================================================
# make_checksums.ps1 - generate SHA256SUMS.txt for a GitHub release.
#
# Reads MANIFEST.json (same folder), collects every asset part the installer
# downloads (per variant), computes its SHA256 and writes the standard
# "hash  filename" lines used by INSTALL.ps1.
#
# Usage:
#   scripts\make_checksums.ps1                     # from staging dir
#   scripts\make_checksums.ps1 -DownloadMissing    # fetch missing parts from the release
#   scripts\make_checksums.ps1 -SplitMissing       # re-split merged archives in staging
#                                                  # (parts were sliced at 700 MiB = 738197504 B)
#   scripts\make_checksums.ps1 -Variant rdna2      # only one family
#
# The generated SHA256SUMS.txt must be uploaded as an asset of the SAME GitHub
# release the parts live on (see docs/RELEASING.md).
# =============================================================================
param(
    [string]$Manifest = "MANIFEST.json",
    [string]$StagingDir = "C:\_release_staging",
    [string]$OutFile = "SHA256SUMS.txt",
    [string]$Owner = "sebastianmechno-sys",
    [string]$Repo = "vllm-rocm-windows-rdna2",
    [string]$Tag = "V2.1",
    [ValidateSet("all","rdna2","rdna3","rdna4")]
    [string]$Variant = "all",
    [long]$ChunkSize = 738197504,
    [switch]$DownloadMissing,
    [switch]$SplitMissing
)
$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $here $Manifest
if (-not (Test-Path $manifestPath)) { $manifestPath = Join-Path (Split-Path -Parent $here) $Manifest }
if (-not (Test-Path $manifestPath)) { Write-Host "ERROR: $Manifest not found" -ForegroundColor Red; exit 1 }

$cfg = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Re-split a merged archive into the .part-XX chunks the release was uploaded with.
function Split-IntoParts {
    param([string]$Merged, [string]$OutName, [string]$Dir)
    if (-not (Test-Path $Merged)) { Write-Host "WARNING: merged archive missing: $Merged" -ForegroundColor Yellow; return $false }
    $n = 1
    $fs = [IO.File]::OpenRead($Merged)
    $buf = New-Object byte[] $ChunkSize
    try {
        for (;;) {
            $read = $fs.Read($buf, 0, $ChunkSize)
            if ($read -eq 0) { break }
            $part = Join-Path $Dir ("{0}.part-{1:D2}" -f $OutName, $n)
            $out = [IO.File]::Create($part)
            try { $out.Write($buf, 0, $read) } finally { $out.Close() }
            Write-Host "  split -> $([IO.Path]::GetFileName($part)) ($read bytes)" -ForegroundColor Cyan
            $n++
            if ($read -lt $ChunkSize) { break }
        }
    } finally { $fs.Close() }
    return $true
}

$files = @()
$splitOnce = @{}   # merged archives already re-split in this run
foreach ($vname in $cfg.variants.PSObject.Properties.Name) {
    if ($Variant -ne "all" -and $vname -ne $Variant) { continue }
    foreach ($a in $cfg.variants.$vname.assets) {
        foreach ($f in $a.files) {
            $path = Join-Path $StagingDir $f
            if (-not (Test-Path $path) -and $SplitMissing -and -not $splitOnce.ContainsKey($a.archive)) {
                $merged = Join-Path $StagingDir $a.archive
                if (Test-Path $merged) {
                    Write-Host "re-splitting $($a.archive) into parts..." -ForegroundColor Yellow
                    # part name = full archive name + .part-XX (e.g. the-rock-venv.tar.zst.part-01)
                    [void](Split-IntoParts -Merged $merged -OutName $a.archive -Dir $StagingDir)
                } else {
                    Write-Host "WARNING: no merged archive for $f (use -DownloadMissing)" -ForegroundColor Yellow
                }
                $splitOnce[$a.archive] = $true
            }
            $files += $f
        }
    }
}

$lines = @()
foreach ($f in ($files | Sort-Object -Unique)) {
    $path = Join-Path $StagingDir $f
    if (-not (Test-Path $path) -and $DownloadMissing) {
        $url = "https://github.com/$Owner/$Repo/releases/download/$Tag/$f"
        Write-Host "fetching missing part: $f" -ForegroundColor Yellow
        & curl.exe -sSL --fail -o $path $url
        if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: could not fetch $f - skipped" -ForegroundColor Yellow; continue }
    }
    if (-not (Test-Path $path)) {
        Write-Host "WARNING: missing in staging (use -DownloadMissing / -SplitMissing): $f" -ForegroundColor Yellow
        continue
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    $lines += "$hash  $f"
    Write-Host "  $hash  $f"
}

if ($lines.Count -eq 0) { Write-Host "ERROR: no parts hashed - nothing to write" -ForegroundColor Red; exit 1 }
$lines | Sort-Object -Unique | Set-Content (Join-Path $here $OutFile) -Encoding ascii
Write-Host "Wrote $($lines.Count) checksums -> $OutFile" -ForegroundColor Green
Write-Host "Upload $OutFile as an asset of the '$Tag' release (web UI: 'Attach binaries')."