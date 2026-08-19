@echo off
rem =============================================================================
rem  INSTALL.bat - one click installer.
rem  Usage:  INSTALL.bat                 (default model + official repo)
rem          INSTALL.bat -Model <owner>/<model>   (pick any public AWQ/GPTQ model)
rem          INSTALL.bat <github-username>   (from your own fork's releases)
rem
rem  Idempotent: checks the machine, downloads only what is missing
rem  (~6 GB from GitHub Releases + Hugging Face), installs everything to the
rem  fixed paths below and runs a verification benchmark at the end.
rem
rem    C:\Python311                        base Python 3.11.9
rem    C:\TheRock\.venv                    torch ROCm venv
rem    C:\TheRock\build\dist\rocm          ROCm runtime libraries
rem    C:\TheRock\ROCM_VLLM_RUNTIME        vLLM + Windows-ROCm plugin
rem    C:\vw_cext_build, C:\vw_hipgemv_build   native GEMV kernels
rem    %USERPROFILE%\.cache\huggingface    model weights
rem
rem Requires Administrator (auto-elevates). Afterwards: VERIFY.bat (test
rem  everything) and CHAT.bat (talk to the model).
rem =============================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL.ps1" %*
echo.
pause
