@echo off
REM Cardiomyocyte Analyzer - one-click start for Windows (double-click this file).
REM First run: creates a Python virtual environment and installs dependencies.
REM Every run: pulls the latest code (if git is installed), starts the server
REM and opens http://localhost:8000 in the browser. Close this window to stop.
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if %errorlevel%==0 if exist ".git" (
  echo [1/4] Pulling latest code (git pull)...
  git pull --ff-only || echo   (git pull failed - check internet or local changes; continuing with current code)
)

cd backend
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py -3
) else (
  where python >nul 2>nul || (
    echo Python was not found. Install Python 3.11+ from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this file again.
    pause
    exit /b 1
  )
  set PY=python
)

if not exist ".venv" (
  echo [2/4] Creating virtual environment (first run only)...
  %PY% -m venv .venv
)
call .venv\Scripts\activate.bat

echo [3/4] Installing/checking packages...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo [4/4] Starting server at http://localhost:8000  (close this window to stop)
start "" "http://localhost:8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
