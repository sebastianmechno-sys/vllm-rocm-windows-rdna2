@echo off
rem =============================================================================
rem  SERVE.bat - start the model server (works like `vllm serve`).
rem  The model becomes an OpenAI-compatible API on http://127.0.0.1:8000
rem  Keep this window open. Talk to it with CHAT.bat (or any OpenAI client).
rem  To change the model: edit config.bat (SERVED_MODEL + MODEL_NAME).
rem  If config.bat is missing (e.g. a fresh ZIP download), the installed
rem  default model is auto-detected from the HuggingFace cache instead.
rem =============================================================================
setlocal

set "VENV_PYTHON=C:\TheRock\.venv\Scripts\python.exe"

if exist "%~dp0config.bat" (
    call "%~dp0config.bat"
    goto :env_ok
)

echo [serve] config.bat not found - auto-detecting the installed model...
if exist "%VENV_PYTHON%" goto :scan_model
echo [serve] ERROR: stack not installed - run INSTALL.bat first.
pause & exit /b 1

:scan_model
set "HF_HUB_DIR=%USERPROFILE%\.cache\huggingface\hub\models--cyankiwi--Qwen3.5-4B-AWQ-4bit\snapshots"
if not exist "%HF_HUB_DIR%" goto :no_model
for /d %%d in ("%HF_HUB_DIR%\*") do (
    if exist "%%d\config.json" set "SERVED_MODEL=%%d"
)
if not defined SERVED_MODEL goto :no_model
set "MODEL_NAME=Qwen3.5-4B-AWQ-4bit"
echo [serve] found %MODEL_NAME% at %SERVED_MODEL%
goto :env_ok

:no_model
echo [serve] ERROR: no model found in the HuggingFace cache.
echo [serve] Run INSTALL.bat first, or create config.bat with SERVED_MODEL + MODEL_NAME.
pause & exit /b 1

:env_ok
if not defined HSA_OVERRIDE_GFX_VERSION set "HSA_OVERRIDE_GFX_VERSION=10.3.1"
set "ROCM_PATH=C:\TheRock\build\dist\rocm"
set "HIP_PATH=C:\TheRock\build\dist\rocm"
set "ROCBLAS_TENSILE_LIBPATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin\rocblas\library"
set "PATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin;%PATH%"
set "PYTHONUTF8=1"
if not defined VLLM_WIN_HIPGEMV_DIR set "VLLM_WIN_HIPGEMV_DIR=C:\vw_hipgemv_build\gemv_w4_hip"

cd /d "C:\TheRock\ROCM_VLLM_RUNTIME\vllm-rocm-windows\run"
rem Guard: if a server is already up on this port, do NOT load a second engine
rem into the GPU (two engines = VRAM exhaustion + weird behavior).
curl -s -o nul -m 2 http://127.0.0.1:%SERVE_PORT%/v1/models >nul 2>&1
if %ERRORLEVEL%==0 (
    echo ============================================================
    echo   [serve] Server ALREADY RUNNING on 127.0.0.1:%SERVE_PORT% - nothing to do.
    echo   Close the other SERVE window first if you want to restart it.
    echo ============================================================
    timeout /t 5 >nul
    exit /b 0
)
echo ============================================================
echo   Starting model server on 127.0.0.1:%SERVE_PORT% ...
echo   Wait for "Application startup complete", then run CHAT.bat
echo   First start takes 8-10 minutes (kernel compilation).
echo ============================================================
"%VENV_PYTHON%" "%~dp0scripts\serve.py"
endlocal