@echo off
set HSA_OVERRIDE_GFX_VERSION=10.3.0
set ROCM_PATH=C:\TheRock\build\dist\rocm
set HIP_PATH=C:\TheRock\build\dist\rocm
set ROCBLAS_TENSILE_LIBPATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin\rocblas\library
set PATH=C:\TheRock\ROCM_VLLM_RUNTIME\bin;%PATH%
echo Motore doppia copia attivo - gfx1031 incluso - originali intatti
cmd /k