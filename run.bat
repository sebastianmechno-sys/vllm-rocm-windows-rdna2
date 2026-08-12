@echo off
REM run.bat - Launch vLLM inference on RX 6750 XT - FIXED FOR WINDOWS ZMQ BUG
setlocal

set THEROCK_ROOT=C:\TheRock
set VENV_PYTHON=%THEROCK_ROOT%\.venv\Scripts\python.exe

REM Critical env - FIX ZMQ ipc:// bug on Windows
set HSA_OVERRIDE_GFX_VERSION=10.3.1
set HIP_VISIBLE_DEVICES=0
set VLLM_TARGET_DEVICE=rocm
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500
set VLLM_ENABLE_V1_MULTIPROCESSING=0
set VLLM_PLUGINS=windows_rocm

REM PATH for TheRock runtime
set PATH=%THEROCK_ROOT%\build\dist\rocm\bin;%THEROCK_ROOT%\build\core\clr\dist\bin;%THEROCK_ROOT%\build\math-libs\BLAS\rocBLAS\build\staging;%PATH%

cd /d %~dp0

echo Starting vLLM on RX 6750 XT with V0 engine (Windows fix)...
echo Model: facebook/opt-125m
echo.

%VENV_PYTHON% inference.py --model facebook/opt-125m --dtype float16 --max-model-len 512 --gpu-memory-utilization 0.5 --attention-backend TRITON_ATTN

pause
