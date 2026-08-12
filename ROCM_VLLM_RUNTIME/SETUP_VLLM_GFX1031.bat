@echo off
REM Setup vLLM build env for gfx1031 - hardest part
REM Esegui dentro C:\TheRock\ROCM_VLLM_RUNTIME\AVVIA.bat

set HSA_OVERRIDE_GFX_VERSION=10.3.0
set PYTORCH_ROCM_ARCH=gfx1031
set HIP_VISIBLE_DEVICES=0
set GPU_ARCH=gfx1031
set VLLM_TARGET_DEVICE=rocm
set VLLM_ATTENTION_BACKEND=TRITON
set MAX_JOBS=8

echo === ENV RDNA2 vLLM ===
echo PYTORCH_ROCM_ARCH=%PYTORCH_ROCM_ARCH%
hipInfo.exe 2>nul || echo hipInfo non in PATH, uso C:\TheRock\build\core\clr\dist\bin\hipInfo.exe
C:\TheRock\build\core\clr\dist\bin\hipInfo.exe

echo.
echo Clono vLLM se non esiste...
if not exist vllm (
  git clone https://github.com/vllm-project/vllm.git
)
cd vllm
git fetch origin
git checkout main
git pull

echo.
echo Creo branch gfx1031...
git checkout -B gfx1031-rdna2

echo.
echo Verifico file attention...
dir csrc\attention\ /b

echo.
echo Fatto. Ora copia il kernel gfx1031 che ti ho dato in:
echo   vllm\csrc\attention\paged_attention_gfx1031.hip
echo Poi prova build test:
echo   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
echo.
cmd /k
