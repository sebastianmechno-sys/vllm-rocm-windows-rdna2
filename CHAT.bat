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
echo [chat] Waiting for the engine to load...
echo [chat] First start can take 8-10 minutes (kernel compilation); later starts ~1 minute.
:wait
timeout /t 5 /nobreak >nul
curl -s -o nul -m 2 %API%
if %ERRORLEVEL%==0 goto :up
echo [chat] ... still loading, server logs are in the other window.
goto :wait

:up
rem Open the chat over a tiny local web server instead of file:// - browsers
rem block API calls from file:// pages, which would leave the chat stuck on
rem "connecting..." forever. scripts\chat_server.py picks a free port, reuses
rem an already-running instance and opens the browser by itself.
set "UI_PY=C:\TheRock\.venv\Scripts\python.exe"
if not exist "%UI_PY%" set "UI_PY=C:\Python311\python.exe"
if exist "%UI_PY%" (
    start "Chat UI web server (keep open)" /MIN "%UI_PY%" "%~dp0scripts\chat_server.py"
) else (
    start "Chat UI web server (keep open)" /MIN python "%~dp0scripts\chat_server.py"
)
endlocal
