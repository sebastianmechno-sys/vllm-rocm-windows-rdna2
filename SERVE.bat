@echo off
rem =============================================================================
rem  SERVE.bat - start the model server (works like `vllm serve`).
rem  The model becomes an OpenAI-compatible API on http://127.0.0.1:8000
rem  Keep this window open. Talk to it with CHAT.bat (or any OpenAI client).
rem  To change the model: edit config.bat (SERVED_MODEL + MODEL_NAME).
rem =============================================================================
setlocal

if exist "%~dp0config.bat" call "%~dp0config.bat"
if not defined VENV_PYTHON set "VENV_PYTHON=C:\TheRock\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto :py_ok
echo [serve] ERROR: stack not installed - run INSTALL.bat first.
pause & exit /b 1
:py_ok

if not defined HSA_OVERRIDE_GFX_VERSION set "HSA_OVERRIDE_GFX_VERSION=10.3.1"
set "ROCM_PATH=C:\TheRock\build\dist\rocm"
set "HIP_PATH=C:\TheRock\build\dist\rocm"
set "ROCBLAS_TENSILE_LIBPATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin\rocblas\library"
set "PATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin;%PATH%"
set "PYTHONUTF8=1"
if not defined VLLM_WIN_HIPGEMV_DIR set "VLLM_WIN_HIPGEMV_DIR=C:\vw_hipgemv_build\gemv_w4_hip"

cd /d "C:\TheRock\ROCM_VLLM_RUNTIME\vllm-rocm-windows\run"
echo ============================================================
echo   Starting model server on 127.0.0.1:8000 ...
echo   Wait for "Application startup complete", then run CHAT.bat
echo ============================================================
"%VENV_PYTHON%" "%~dp0scripts\serve.py"
endlocal
