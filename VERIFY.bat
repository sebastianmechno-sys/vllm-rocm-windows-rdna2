@echo off
rem =============================================================================
rem  VERIFY.bat - one-click verification of the whole stack:
rem    [1/3] ROCm GPU detection (HIP version, GPU name, VRAM)
rem    [2/3] GPU raw FP16 power (4096^3 GEMM via native rocblas-bench.exe)
rem    [3/3] vLLM tuned decode benchmark (overall_tok_s)
rem  Expects ~26 TFLOPS and ~59-62 tok/s on an RX 6750 XT.
rem =============================================================================
setlocal

if exist "%~dp0config.bat" call "%~dp0config.bat"
if not defined VENV_PYTHON set "VENV_PYTHON=C:\TheRock\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto :py_ok
echo [verify] ERROR: stack not installed - run INSTALL.bat first.
pause & exit /b 1
:py_ok

set "HSA_OVERRIDE_GFX_VERSION=10.3.1"
set "ROCM_PATH=C:\TheRock\build\dist\rocm"
set "HIP_PATH=C:\TheRock\build\dist\rocm"
set "ROCBLAS_TENSILE_LIBPATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin\rocblas\library"
set "PATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin;%PATH%"
set "PYTHONUTF8=1"

echo ============================================================
echo   [1/3] ROCm GPU detection
echo ============================================================
"%VENV_PYTHON%" -c "import torch; print('HIP runtime:', torch.version.hip); print('GPU visible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); p=torch.cuda.get_device_properties(0); print(f'VRAM: {p.total_memory/1073741824:.1f} GB - Compute units: {p.multi_processor_count}')"
if errorlevel 1 ( echo [1/3] FAILED & pause & exit /b 1 )

echo.
echo ============================================================
echo   [2/3] GPU raw FP16 power ^(native rocblas-bench, 4096^3 GEMM^)
echo   RX 6750 XT reference: ~26 TFLOPS ^(25900+ Gflops^)
echo ============================================================
"C:\TheRock\ROCM_VLLM_RUNTIME\bin\rocblas-bench\rocblas-bench.exe" -f gemm -r f16_r -m 4096 -n 4096 -k 4096 -i 10
if errorlevel 1 ( echo [2/3] FAILED & pause & exit /b 1 )

echo.
echo ============================================================
echo   [3/3] vLLM tuned decode benchmark ^(expect ~59-62 tok/s^)
echo ============================================================
if not defined BENCH_MODEL (
    echo ERROR: no model configured - run INSTALL.bat first.
    pause & exit /b 1
)
set "VLLM_WIN_BF16_GEMV=1"
set "VLLM_WIN_HIPGEMV=1"
set "BENCH_MODE=graph"
if not defined BENCH_MAXTOK set "BENCH_MAXTOK=512"
set "BENCH_MODEL=%BENCH_MODEL%"
cd /d "C:\TheRock\ROCM_VLLM_RUNTIME\vllm-rocm-windows\run"
"%VENV_PYTHON%" "%~dp0scripts\benchmark.py"

echo.
echo ============================================================
echo   ALL CHECKS DONE - read overall_tok_s above.
echo ============================================================
pause
endlocal
