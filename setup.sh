#!/bin/bash
# ============================================================
# EmotiSign — Linux/Mac Setup Script
# Run once after cloning: chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo ""
echo "========================================"
echo " EmotiSign Setup"
echo "========================================"
echo ""

# ── Check Python ──────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+ from https://python.org"
    exit 1
fi
echo "[OK] $(python3 --version)"

# ── Check Node.js ─────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo "[ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org"
    exit 1
fi
echo "[OK] $(node --version)"

# ── Backend setup ─────────────────────────────────────────
echo ""
echo "[1/4] Setting up Python virtual environment..."
cd emotisign-backend
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "[OK] Python dependencies installed"

# ── Create .env if missing ────────────────────────────────
if [ ! -f .env ]; then
    echo "[3/4] Creating .env from template..."
    cp .env.example .env
    echo "[NOTE] .env created. Edit SECRET_KEY before deploying to production."
else
    echo "[3/4] .env already exists, skipping."
fi

# ── Create required directories ───────────────────────────
mkdir -p uploads/videos uploads/audio uploads/tts_output static/signs

cd ..

# ── Frontend setup ────────────────────────────────────────
echo "[4/4] Installing Node.js dependencies..."
cd emotisign-frontend
npm install --silent

if [ ! -f .env.local ]; then
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
    echo "[OK] Frontend .env.local created"
fi

cd ..

echo ""
echo "========================================"
echo " Setup complete!"
echo " Run: ./start.sh  to launch the app"
echo "========================================"
echo ""
