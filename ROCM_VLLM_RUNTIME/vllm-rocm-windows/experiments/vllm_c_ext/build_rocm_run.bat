@echo off
call "%~dp0..\..\tools\winrocm_env.bat" || exit /b 1
python -u "%~dp0build_rocm_c.py" %*
