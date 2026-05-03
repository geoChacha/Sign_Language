@echo off
REM ============================================================
REM EmotiSign — Start both backend and frontend
REM ============================================================

echo.
echo ========================================
echo  Starting EmotiSign
echo ========================================
echo.
echo  Backend  → http://localhost:8000
echo  Frontend → http://localhost:3000
echo  API Docs → http://localhost:8000/docs
echo.
echo  Press Ctrl+C in each window to stop.
echo ========================================
echo.

REM ── Start backend in a new window ─────────────────────────
start "EmotiSign Backend" cmd /k "cd /d %~dp0emotisign-backend && call venv\Scripts\activate.bat && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

REM ── Wait 3 seconds for backend to start ───────────────────
timeout /t 3 /nobreak >nul

REM ── Start frontend in a new window ────────────────────────
start "EmotiSign Frontend" cmd /k "cd /d %~dp0emotisign-frontend && npm run dev"

echo Both servers are starting in separate windows.
echo.
pause
