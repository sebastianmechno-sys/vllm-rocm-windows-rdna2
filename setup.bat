@echo off
REM setup.bat - One-time setup for vLLM ROCm RDNA2 on Windows 11
echo [1/4] Setting ROCm paths...
set THEROCK_ROOT=C:\TheRock
set ROCM_BIN=%THEROCK_ROOT%\build\core\clr\dist\bin
set ROCBLAS_BIN=%THEROCK_ROOT%\build\math-libs\BLAS\rocBLAS\build\clients\staging
set VENV_PYTHON=%THEROCK_ROOT%\.venv\Scripts\python.exe
set PATH=%THEROCK_ROOT%\build\dist\rocm\bin;%ROCM_BIN%;%ROCBLAS_BIN%;%PATH%
setx PATH "%THEROCK_ROOT%\build\dist\rocm\bin;%ROCM_BIN%;%ROCBLAS_BIN%;%PATH%" >nul

echo [2/4] Setting GPU overrides for RX 6750 XT gfx1031...
set HSA_OVERRIDE_GFX_VERSION=10.3.1
set HIP_VISIBLE_DEVICES=0
set VLLM_TARGET_DEVICE=rocm
set VLLM_ENABLE_V1_MULTIPROCESSING=0
setx HSA_OVERRIDE_GFX_VERSION 10.3.1 >nul
setx HIP_VISIBLE_DEVICES 0 >nul
setx VLLM_TARGET_DEVICE rocm >nul
setx VLLM_ENABLE_V1_MULTIPROCESSING 0 >nul

echo [3/4] Setting distributed defaults...
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500
setx MASTER_ADDR 127.0.0.1 >nul
setx MASTER_PORT 29500 >nul

echo [4/4] Verifying...
where rocblas-bench.exe
%VENV_PYTHON% -c "import torch; print(f'torch {torch.__version__} | cuda_avail {torch.cuda.is_available()} | dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Setup done. Now run run.bat
pause
