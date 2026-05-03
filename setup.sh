#!/bin/bash
# ============================================================
# EmotiSign — Linux/Mac Setup Script
# Run once after cloning: chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

REQUIRED_PYTHON="3.10.9"
REQUIRED_PYTHON_MINOR="3.10"

echo ""
echo "========================================"
echo " EmotiSign Setup"
echo "========================================"
echo ""

# ── Check / Install Python 3.10 ───────────────────────────
PYTHON_EXE=""

# Try common Python 3.10 locations
for cmd in python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oP '3\.10\.\d+' || true)
        if [ -n "$ver" ]; then
            PYTHON_EXE="$cmd"
            echo "[OK] Found Python $ver at $(which $cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_EXE" ]; then
    echo "[INFO] Python 3.10 not found. Attempting to install..."
    echo ""

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS — use Homebrew
        if command -v brew &>/dev/null; then
            echo "Installing Python 3.10 via Homebrew..."
            brew install python@3.10
            PYTHON_EXE="python3.10"
        else
            echo "[ERROR] Homebrew not found. Install Python 3.10.9 manually:"
            echo "        https://www.python.org/ftp/python/3.10.9/python-3.10.9-macos11.pkg"
            echo "        Or install Homebrew first: https://brew.sh"
            exit 1
        fi
    elif command -v apt-get &>/dev/null; then
        # Debian/Ubuntu
        echo "Installing Python 3.10 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
        PYTHON_EXE="python3.10"
    elif command -v dnf &>/dev/null; then
        # Fedora/RHEL
        echo "Installing Python 3.10 via dnf..."
        sudo dnf install -y python3.10 python3.10-devel
        PYTHON_EXE="python3.10"
    else
        echo "[ERROR] Cannot auto-install Python 3.10. Please install manually:"
        echo "        https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz"
        exit 1
    fi

    echo "[OK] Python 3.10 installed"
fi

# ── Check Node.js ─────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo "[ERROR] Node.js not found. Install Node.js 18+ from https://nodejs.org"
    exit 1
fi
echo "[OK] Node.js $(node --version) found"

# ── Backend setup ─────────────────────────────────────────
echo ""
echo "[1/4] Setting up Python virtual environment..."
cd emotisign-backend
"$PYTHON_EXE" -m venv venv
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
