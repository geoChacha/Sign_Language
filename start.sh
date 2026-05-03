#!/bin/bash
# ============================================================
# EmotiSign — Start both backend and frontend
# ============================================================

echo ""
echo "========================================"
echo " Starting EmotiSign"
echo "========================================"
echo " Backend  → http://localhost:8000"
echo " Frontend → http://localhost:3000"
echo " API Docs → http://localhost:8000/docs"
echo ""
echo " Press Ctrl+C to stop all servers"
echo "========================================"
echo ""

# Cleanup on exit
trap 'echo ""; echo "Stopping servers..."; kill $(jobs -p) 2>/dev/null; exit' INT TERM

# ── Start backend ─────────────────────────────────────────
cd emotisign-backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# ── Wait for backend to start ─────────────────────────────
sleep 3

# ── Start frontend ────────────────────────────────────────
cd emotisign-frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Both servers running. Press Ctrl+C to stop."
wait
