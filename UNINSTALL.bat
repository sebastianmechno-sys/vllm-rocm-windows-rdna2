@echo off
rem =============================================================================
rem  UNINSTALL.bat - clean removal of the installed stack.
rem  Removes C:\TheRock, the native kernel builds and config.bat (and, if you
rem  confirm, C:\Python311 and the HuggingFace model cache). The repo folder
rem  itself is kept - delete it manually if you no longer need it.
rem =============================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0UNINSTALL.ps1"
echo.
pause