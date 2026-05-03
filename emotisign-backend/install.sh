#!/bin/bash
# Installation script for EmotiSign Backend (Linux/Mac)
# This script installs all required Python packages

echo "============================================================"
echo "EmotiSign Backend - Installation Script"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9 or 3.10"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip is not installed"
    echo "Please install pip"
    exit 1
fi

echo "pip found:"
pip3 --version
echo ""

# Upgrade pip
echo "============================================================"
echo "Step 1: Upgrading pip..."
echo "============================================================"
python3 -m pip install --upgrade pip
echo ""

# Install CPU-only PyTorch first (faster, works on all systems)
echo "============================================================"
echo "Step 2: Installing PyTorch (CPU version)..."
echo "============================================================"
echo "Note: For GPU support, see README.md for CUDA installation"
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
echo ""

# Install all other dependencies
echo "============================================================"
echo "Step 3: Installing other dependencies..."
echo "============================================================"
pip3 install fastapi==0.115.5
pip3 install uvicorn[standard]==0.32.1
pip3 install python-multipart==0.0.12
pip3 install python-jose[cryptography]==3.3.0
pip3 install passlib[bcrypt]==1.7.4
pip3 install bcrypt==4.2.1
pip3 install sqlalchemy==2.0.36
pip3 install aiosqlite==0.20.0
pip3 install pydantic[email]==2.10.3
pip3 install pydantic-settings==2.6.1
pip3 install aiofiles==24.1.0
pip3 install mediapipe==0.9.3
pip3 install opencv-python==4.8.1.78
pip3 install numpy==1.24.3
pip3 install pytest==7.4.3
pip3 install pytest-asyncio==0.21.1
pip3 install pytest-cov==4.1.0
pip3 install requests==2.31.0
echo ""

# Verify installation
echo "============================================================"
echo "Step 4: Verifying installation..."
echo "============================================================"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python3 -c "import sqlalchemy; print('SQLAlchemy:', sqlalchemy.__version__)"
echo ""

echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Run health check: python3 health_check.py"
echo "2. Run tests: python3 run_tests.py"
echo "3. Start server: uvicorn main:app --reload"
echo ""
