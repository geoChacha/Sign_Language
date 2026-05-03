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

REM ── Check Python ──────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

REM ── Check Node.js ─────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js found

REM ── Backend setup ─────────────────────────────────────────
echo.
echo [1/4] Setting up Python virtual environment...
cd emotisign-backend
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies...
pip install --upgrade pip --quiet
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
