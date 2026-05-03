@echo off
REM Quick Start Script for EmotiSign Backend (Windows)
REM This script installs dependencies, runs tests, and starts the server

echo ============================================================
echo EmotiSign Backend - Quick Start
echo ============================================================
echo.

REM Step 1: Install dependencies
echo Step 1/4: Installing dependencies...
call install.bat
if errorlevel 1 (
    echo Installation failed!
    pause
    exit /b 1
)

REM Step 2: Run health check
echo.
echo Step 2/4: Running health check...
python health_check.py
if errorlevel 1 (
    echo Health check failed!
    pause
    exit /b 1
)

REM Step 3: Run tests
echo.
echo Step 3/4: Running tests...
python run_tests.py
if errorlevel 1 (
    echo Tests failed!
    pause
    exit /b 1
)

REM Step 4: Start server
echo.
echo Step 4/4: Starting server...
echo.
echo ============================================================
echo Server will start on http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Press CTRL+C to stop the server
echo ============================================================
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
