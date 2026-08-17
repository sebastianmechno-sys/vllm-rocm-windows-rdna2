@echo off
rem =============================================================================
rem  CHAT.bat - open the web chat with the model.
rem  Requires the server running (SERVE.bat). If it is not up, we start it.
rem =============================================================================
setlocal
set "API=http://127.0.0.1:8000/v1/models"

curl -s -o nul -m 2 %API%
if %ERRORLEVEL%==0 goto :up
echo [chat] Server not running - starting it in a new window...
start "" "%~dp0SERVE.bat"
echo [chat] Waiting for the engine to load (about 1 minute)...
:wait
timeout /t 5 /nobreak >nul
curl -s -o nul -m 2 %API%
if %ERRORLEVEL%==0 goto :up
echo [chat] ... still loading, server logs are in the other window.
goto :wait

:up
start "" "%~dp0chat.html"
endlocal
