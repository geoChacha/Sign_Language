@echo off
REM ============================================================
REM EmotiSign — Windows Setup Script
REM Run once after cloning: setup.bat
REM ============================================================

echo.
echo ========================================
echo  EmotiSign Setup
echo ========================================
echo.

REM ── Check / Install Python 3.10.9 ─────────────────────────
set PYTHON_VERSION=3.10.9
set PYTHON_INSTALLER=python-3.10.9-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
set PYTHON_EXE=

REM Try py launcher first (most reliable on Windows)
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_EXE=py -3.10
    echo [OK] Python 3.10 found via py launcher
    goto :python_ok
)

REM Try python3.10 directly
python3.10 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_EXE=python3.10
    echo [OK] Python 3.10 found
    goto :python_ok
)

REM Try python and check version
python --version 2>&1 | findstr /C:"3.10" >nul
if not errorlevel 1 (
    set PYTHON_EXE=python
    echo [OK] Python 3.10 found
    goto :python_ok
)

REM Python 3.10 not found — download and install it
echo [INFO] Python 3.10.9 not found. Downloading installer...
echo        URL: %PYTHON_URL%
echo.

REM Check if curl is available (Windows 10+)
curl --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] curl not found. Please install Python 3.10.9 manually:
    echo         %PYTHON_URL%
    echo         Then re-run this script.
    pause
    exit /b 1
)

echo Downloading Python 3.10.9...
curl -L -o "%TEMP%\%PYTHON_INSTALLER%" "%PYTHON_URL%"
if errorlevel 1 (
    echo [ERROR] Download failed. Please install Python 3.10.9 manually:
    echo         %PYTHON_URL%
    pause
    exit /b 1
)

echo Installing Python 3.10.9 (this may take a minute)...
REM /quiet = silent install, PrependPath=1 = add to PATH, Include_pip=1 = include pip
"%TEMP%\%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1
if errorlevel 1 (
    echo [ERROR] Python installation failed. Please install manually:
    echo         %PYTHON_URL%
    pause
    exit /b 1
)

echo [OK] Python 3.10.9 installed successfully
echo [NOTE] Please close and reopen this terminal, then run setup.bat again.
pause
exit /b 0

:python_ok

REM ── Check Node.js ─────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org
    echo         Then re-run this script.
    pause
    exit /b 1
)
echo [OK] Node.js %node_ver% found

REM ── Backend setup ─────────────────────────────────────────
echo.
echo [1/4] Setting up Python virtual environment...
cd emotisign-backend

REM Create venv using the correct Python 3.10
if "%PYTHON_EXE%"=="py -3.10" (
    py -3.10 -m venv venv
) else (
    %PYTHON_EXE% -m venv venv
)

call venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Python dependencies installed

REM ── Create .env if missing ────────────────────────────────
if not exist .env (
    echo [3/4] Creating .env from template...
    copy .env.example .env
    echo [NOTE] .env created. Edit SECRET_KEY before deploying to production.
) else (
    echo [3/4] .env already exists, skipping.
)

REM ── Create required directories ───────────────────────────
if not exist uploads\videos mkdir uploads\videos
if not exist uploads\audio mkdir uploads\audio
if not exist uploads\tts_output mkdir uploads\tts_output
if not exist static\signs mkdir static\signs

cd ..

REM ── Frontend setup ────────────────────────────────────────
echo [4/4] Installing Node.js dependencies...
cd emotisign-frontend
npm install --silent
if errorlevel 1 (
    echo [ERROR] Failed to install Node.js dependencies
    pause
    exit /b 1
)

REM ── Create frontend .env.local if missing ─────────────────
if not exist .env.local (
    echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local
    echo [OK] Frontend .env.local created
)

cd ..

echo.
echo ========================================
echo  Setup complete!
echo  Run: start.bat  to launch the app
echo ========================================
echo.
pause
