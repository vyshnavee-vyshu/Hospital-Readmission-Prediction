@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or later and add it to PATH.
  pause
  exit /b 1
)

python --version
if errorlevel 1 (
  echo Unable to run Python.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo.
echo Starting Hospital Readmission Prediction at http://127.0.0.1:8000
echo Stop the server with Ctrl+C
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
