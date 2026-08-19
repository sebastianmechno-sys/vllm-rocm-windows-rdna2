# =============================================================================
# INSTALL.ps1 - one-click installer for vllm-rocm-windows-rdna2-oneclick
#
# Reads MANIFEST.json (shipped with the repo) to know the exact release asset
# names, downloads them from GitHub Releases, extracts to the fixed paths the
# stack expects, installs base Python, fetches the 4-bit model from Hugging
# Face, then verifies everything with a short benchmark.
#
# Final layout:
#   C:\Python311                                   base Python 3.11.9
#   C:\TheRock\.venv                               torch ROCm + venv
#   C:\TheRock\build\dist\rocm                     ROCm runtime
#   C:\TheRock\ROCM_VLLM_RUNTIME                   vLLM + plugin + ROCm bin
#   C:\vw_cext_build, C:\vw_hipgemv_build          native kernels
#   HF cache: cyankiwi--Qwen3.5-4B-AWQ-4bit        model weights
# =============================================================================
param(
    [string]$Owner = "sebastianmechno-sys",
    [string]$Repo  = "vllm-rocm-windows-rdna2",
    [string]$Tag   = "V2.0",
    [string]$BaseUrl = "",
    [string]$Prefix = "",
    [string]$Model = "cyankiwi/Qwen3.5-4B-AWQ-4bit",
    [switch]$SkipModel
)
$ErrorActionPreference = 'Stop'

# -------------------------------------------------------- self-elevate -------
$isAdmin = (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[setup] requesting Administrator rights (UAC prompt)..." -ForegroundColor Yellow
    $sb = New-Object System.Text.StringBuilder
    foreach ($k in $PSBoundParameters.Keys) {
        $v = $PSBoundParameters[$k]
        if ($v -is [bool]) { if ($v) { [void]$sb.Append(" -$k") } }
        else { [void]$sb.Append(" -$k `"$v`"") }
    }
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",("`"{0}`"" -f $MyInvocation.MyCommand.Path),$sb.ToString()
    exit
}

$stage  = "C:\_release_staging"
$here   = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($BaseUrl) { $base = $BaseUrl } else { $base = "https://github.com/$Owner/$Repo/releases/download/$Tag" }

# Sandbox support: set -Prefix <dir> (or VLLM_SETUP_PREFIX) to install
# everything under <dir> instead of the real fixed paths (isolated testing).
$P = $Prefix
if (-not $P) { $P = $env:VLLM_SETUP_PREFIX }
if (-not $P) { $P = "" }
$pyRoot     = if ($P) { "$P\Python311" } else { "C:\Python311" }
$stackRoot  = if ($P) { "$P\TheRock" } else { "C:\TheRock" }
$kVwCext    = if ($P) { "$P\vw_cext_build" } else { "C:\vw_cext_build" }
$kVwHip     = if ($P) { "$P\vw_hipgemv_build" } else { "C:\vw_hipgemv_build" }
$manifest = Join-Path $here "MANIFEST.json"
if (-not (Test-Path $manifest)) {
    Write-Host "[setup] ERROR: MANIFEST.json missing next to INSTALL.ps1" -ForegroundColor Red; exit 1
}

function Log($m)  { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Err($m)  { Write-Host "[setup] ERROR: $m" -ForegroundColor Red; exit 1 }

$cfg = Get-Content $manifest -Raw | ConvertFrom-Json

# ------------------------------------------------------------- preflight ----
$free = (Get-PSDrive C).Free / 1GB
if ($free -lt 25) { Err "need ~25 GB free on C: (have $([math]::Round($free,1)) GB)" }
New-Item -ItemType Directory -Force $stage | Out-Null
Start-Transcript "$stage\setup_transcript.log" -Force | Out-Null

# GPU check: RDNA2 family (gfx1030/1031/1032) is what this stack was built for.
$gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
Log "GPU(s): $($gpus -join ', ')"
$rdna2 = $gpus | Where-Object { $_ -match 'RX 6[4-9]\d0|Radeon 6[4-9]0?M|Radeon Pro V620' }
if (-not $rdna2) {
    Write-Host "[setup] WARNING: no RDNA2 (RX 6600/6700/6800/6900 series / Radeon Pro V620) GPU detected." -ForegroundColor Yellow
    Write-Host "[setup] The stack may still work on other AMD GPUs via HSA_OVERRIDE, but it is" -ForegroundColor Yellow
    Write-Host "[setup] only validated on RDNA2. Proceeding anyway in 10s (Ctrl+C to abort)..." -ForegroundColor Yellow
    Start-Sleep 10
}

function Dl($url, $dst) {
    if (Test-Path $dst) { Log "  cached: $dst"; return }
    Log "  get $([IO.Path]::GetFileName($dst))"
    & curl.exe -sSL --fail --retry 3 --retry-delay 3 -C - -o "$dst.part" $url
    if ($LASTEXITCODE -ne 0 -and -not (Test-Path "$dst.part")) {
        Invoke-WebRequest -Uri $url -OutFile "$dst.part" -UseBasicParsing
    }
    if ($LASTEXITCODE -ne 0 -and -not (Test-Path "$dst.part")) { Err "download failed: $url" }
    Move-Item "$dst.part" $dst -Force
}

# ------------------------------------------------------------ 1. python -----
if (-not (Test-Path "$pyRoot\python.exe")) {
    Log "1/6 installing Python 3.11.9 -> $pyRoot"
    $pyInst = Join-Path $stage "python-3.11.9-amd64.exe"
    Dl "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" $pyInst
    $ip = Start-Process $pyInst -ArgumentList "/quiet","InstallAllUsers=1","TargetDir=$pyRoot","Include_pip=1","Include_launcher=0","Include_test=0" -Wait -PassThru
    if (-not (Test-Path "$pyRoot\python.exe")) {
        Log "  MSI installer failed (exit $($ip.ExitCode)) - falling back to NuGet zip"
        $nup = Join-Path $stage "python-3.11.9.nupkg"
        Dl "https://www.nuget.org/api/v2/package/python/3.11.9" $nup
        $nx = Join-Path $stage "nuget_py"
        New-Item -ItemType Directory -Force $nx | Out-Null
        & tar -xf $nup -C $nx
        New-Item -ItemType Directory -Force $pyRoot | Out-Null
        Copy-Item "$nx\tools\*" $pyRoot -Recurse -Force
    }
    if (-not (Test-Path "$pyRoot\python.exe")) { Err "Python install failed (MSI and NuGet fallback)" }
    Log "  python OK: $(& $pyRoot\python.exe --version)"
} else { Log "1/6 Python already at $pyRoot" }

# -------------------------------------------------------- 2..4. archives ----
foreach ($a in $cfg.assets) {
    $dstDir = $a.extract_to
    if ($P) { $dstDir = $P + $a.extract_to.Substring(2) }
    # skip if a marker file says this archive is already extracted (prefix-aware)
    $sfx = if ($P) { ("." + ($P -replace '[\\:]','_')) } else { "" }
    $marker = Join-Path $stage ("done." + [IO.Path]::GetFileNameWithoutExtension($a.archive) + $sfx)
    if ((Test-Path $dstDir) -and (Test-Path $marker)) {
        Log "2/6 $([IO.Path]::GetFileName($a.archive)) already extracted"; continue
    }
    Log "2/6 $([IO.Path]::GetFileName($a.archive)) -> $dstDir"
    $parts = @()
    foreach ($f in $a.files) {
        $local = Join-Path $stage $f
        Dl "$base/$f" $local
        $parts += $local
    }
    $arc = Join-Path $stage $a.archive
    if ($parts.Count -eq 1 -and $parts[0] -eq $arc) { $merged = $arc }
    else {
        Log "  joining $($parts.Count) parts"
        $out = [IO.File]::Create($arc)
        foreach ($partFile in $parts) { $in = [IO.File]::OpenRead($partFile); $in.CopyTo($out); $in.Close() }
        $out.Close()
    }
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Force $dstDir | Out-Null }
    & tar --zstd -xf $arc -C $dstDir
    if ($LASTEXITCODE -ne 0) { Err "extraction failed: $arc" }
    Set-Content $marker (Get-Date).ToString('s')
    Remove-Item $arc -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------- 3. venv fix -----
Log "3/6 pointing the venv at $pyRoot"
$venvCfg = "$stackRoot\.venv\pyvenv.cfg"
if (-not (Test-Path $venvCfg)) { Err "venv archive broken (no pyvenv.cfg)" }
Set-Content $venvCfg @"
home = $pyRoot
include-system-site-packages = false
version = 3.11.9
executable = $pyRoot\python.exe
command = $pyRoot\python.exe -m venv $stackRoot\.venv
"@
$venvPy = "$stackRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Err "venv python not found after extract" }
$oldEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $venvPy -c "import torch; assert torch.version.hip; print('torch', torch.__version__, 'hip', torch.version.hip)" > "$stage\check.txt" 2>&1
$ErrorActionPreference = $oldEAP
if ($LASTEXITCODE -ne 0) { Get-Content "$stage\check.txt"; Err "torch self-check failed" }
Get-Content "$stage\check.txt"

# ------------------------------------------------------------- 4. model -----
$modelDir = ""
if (-not $SkipModel) {
    $modelShort = ($Model -split '/')[-1]
    # fast path: local HF cache hit -> no network needed
    $hubRoot = Join-Path $env:USERPROFILE (".cache\huggingface\hub\models--" + ($Model -replace '/', '--') + "\snapshots")
    if (Test-Path $hubRoot) {
        $snap = Get-ChildItem $hubRoot -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path (Join-Path $_.FullName "model-00001-of-00001.safetensors") } | Select-Object -First 1
        if ($snap) { $modelDir = $snap.FullName }
    }
    if ($modelDir) {
        Log "4/6 model already in HF cache (no download): $modelDir"
    } else {
        Log "4/6 downloading model from Hugging Face: $Model"
        $oldEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'   # huggingface_hub logs to stderr; EAP=Stop would kill the script
        & $venvPy -c "from huggingface_hub import snapshot_download; print(snapshot_download('$Model', token=False))" > "$stage\model_path.txt" 2> "$stage\model_err.log"
        $ErrorActionPreference = $oldEAP
        $modelLine = Get-Content "$stage\model_path.txt" -ErrorAction SilentlyContinue | Select-Object -Last 1
        $modelDir = if ($modelLine) { $modelLine.ToString().Trim() } else { "" }
        if (-not $modelDir -or -not (Test-Path $modelDir)) {
            Get-Content "$stage\model_err.log" -ErrorAction SilentlyContinue | Select-Object -Last 5
            Err "model download failed"
        }
        Log "  model at $modelDir"
    }
} else { Log "4/6 skipped (-SkipModel)" }

# -------------------------------------------------------- 5. bench config ---
if ($modelDir) {
@"
@echo off
rem generated by INSTALL.ps1 - edit this file to use a different model
rem SERVED_MODEL = folder containing the model (config.json + safetensors)
rem MODEL_NAME   = name shown in the chat and used by the API
set "VENV_PYTHON=$venvPy"
set "BENCH_MODEL=$modelDir"
set "SERVED_MODEL=$modelDir"
set "MODEL_NAME=$modelShort"
"@ | Set-Content "$here\config.bat"
Log "5/6 wrote config.bat"
} else {
    Log "5/6 skipped config.bat (-SkipModel: set BENCH_MODEL manually, then run VERIFY.bat)"
}

# ----------------------------------------------------------- 6. verify ------
if (-not $modelDir) { Log "6/6 skipped verification (-SkipModel)"; Log "SETUP COMPLETE"; exit 0 }
Log "6/6 verification benchmark (128 tokens, expect ~55-60 tok/s on RX 6750 XT)"
$env:HSA_OVERRIDE_GFX_VERSION   = "10.3.1"
$env:ROCM_PATH                  = "$stackRoot\build\dist\rocm"
$env:HIP_PATH                   = "$stackRoot\build\dist\rocm"
$env:ROCBLAS_TENSILE_LIBPATH    = "$stackRoot\ROCM_VLLM_RUNTIME\bin\rocblas\library"
$env:PATH                       = "$stackRoot\ROCM_VLLM_RUNTIME\bin;$env:PATH"
$env:VLLM_WIN_BF16_GEMV         = "1"
$env:VLLM_WIN_HIPGEMV           = "1"
$env:BENCH_MODE                 = "graph"
$env:BENCH_MAXTOK               = "128"
$env:BENCH_MODEL                = $modelDir
Push-Location "$stackRoot\ROCM_VLLM_RUNTIME\vllm-rocm-windows\run"
$oldEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'   # vllm logs many warnings to stderr
& $venvPy "$here\scripts\benchmark.py" 2>&1 | Tee-Object "$here\results\setup_verify.log" | Select-String "overall_tok_s|prompt_tokens"
$ErrorActionPreference = $oldEAP
Pop-Location
Log "SETUP COMPLETE - run VERIFY.bat for the full verification, CHAT.bat to chat."
