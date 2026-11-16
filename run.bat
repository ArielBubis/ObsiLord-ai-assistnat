@echo off
rem Start Obsi-Lord and open its page in the default browser.
rem The server listens on 127.0.0.1 only. Close this window or press Ctrl+C to stop it.
setlocal
cd /d "%~dp0"

if "%~1"=="--open-when-ready" goto open_when_ready

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No virtual environment found at .venv
    echo Create it with:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo   .venv\Scripts\pip install -e .
    pause
    exit /b 1
)
if not exist config.toml (
    echo config.toml not found. Copy config.example.toml to config.toml and edit it.
    pause
    exit /b 1
)

rem The port comes from [server] in config.toml.
set "PORT="
for /f "usebackq" %%p in (`%PY% -c "from obsilord.config import load_settings; print(load_settings().server.port)" 2^>nul`) do set "PORT=%%p"
if not defined PORT (
    echo Could not read config.toml. The error is:
    "%PY%" -c "from obsilord.config import load_settings; load_settings()"
    pause
    exit /b 1
)

rem A second copy of this script waits for the server, then opens the page.
start "" /b cmd /c ""%~f0" --open-when-ready %PORT%"

echo Starting Obsi-Lord on http://127.0.0.1:%PORT%
"%PY%" -m obsilord
if errorlevel 1 pause
exit /b

:open_when_ready
rem Up to 60 tries, one second apart.
set "URL=http://127.0.0.1:%~2/"
for /l %%i in (1,1,60) do (
    curl -s -o nul --max-time 2 "%URL%" && (
        start "" "%URL%"
        exit /b 0
    )
    ping -n 2 127.0.0.1 >nul
)
exit /b 1
