@echo off
REM Installation script for EmotiSign Backend (Windows)
REM This script installs all required Python packages

echo ============================================================
echo EmotiSign Backend - Installation Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or 3.10 from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Please install pip
    pause
    exit /b 1
)

echo pip found:
pip --version
echo.

REM Upgrade pip
echo ============================================================
echo Step 1: Upgrading pip...
echo ============================================================
python -m pip install --upgrade pip
echo.

REM Install CPU-only PyTorch first (faster, works on all systems)
echo ============================================================
echo Step 2: Installing PyTorch (CPU version)...
echo ============================================================
echo Note: For GPU support, see README.md for CUDA installation
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
echo.

REM Install all other dependencies
echo ============================================================
echo Step 3: Installing other dependencies...
echo ============================================================
pip install fastapi==0.115.5
pip install uvicorn[standard]==0.32.1
pip install python-multipart==0.0.12
pip install python-jose[cryptography]==3.3.0
pip install passlib[bcrypt]==1.7.4
pip install bcrypt==4.2.1
pip install sqlalchemy==2.0.36
pip install aiosqlite==0.20.0
pip install pydantic[email]==2.10.3
pip install pydantic-settings==2.6.1
pip install aiofiles==24.1.0
pip install mediapipe==0.9.3
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install pytest-cov==4.1.0
pip install requests==2.31.0
echo.

REM Verify installation
echo ============================================================
echo Step 4: Verifying installation...
echo ============================================================
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import sqlalchemy; print('SQLAlchemy:', sqlalchemy.__version__)"
echo.

echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Run health check: python health_check.py
echo 2. Run tests: python run_tests.py
echo 3. Start server: uvicorn main:app --reload
echo.
pause
