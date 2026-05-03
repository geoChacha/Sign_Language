@echo off
REM ============================================================
REM EmotiSign Backend - Virtual Environment Setup (Windows)
REM ============================================================
REM This script creates a fresh Python virtual environment
REM and installs all dependencies in the correct order
REM ============================================================

echo.
echo ============================================================
echo EmotiSign Backend - Virtual Environment Setup
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

REM Remove old virtual environment if it exists
if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
    echo.
)

REM Create new virtual environment
echo ============================================================
echo Step 1: Creating virtual environment...
echo ============================================================
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

REM Activate virtual environment
echo ============================================================
echo Step 2: Activating virtual environment...
echo ============================================================
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Upgrade pip
echo ============================================================
echo Step 3: Upgrading pip...
echo ============================================================
python -m pip install --upgrade pip
echo.

REM Install NumPy first (critical for compatibility)
echo ============================================================
echo Step 4: Installing NumPy (foundation package)...
echo ============================================================
pip install numpy==1.26.4
if errorlevel 1 (
    echo ERROR: Failed to install NumPy
    pause
    exit /b 1
)
echo NumPy installed successfully!
echo.

REM Install PyTorch (CPU version)
echo ============================================================
echo Step 5: Installing PyTorch (CPU version)...
echo ============================================================
echo Note: For GPU support, see README.md for CUDA installation
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo PyTorch installed successfully!
echo.

REM Install MediaPipe and OpenCV (compatible versions)
echo ============================================================
echo Step 6: Installing MediaPipe and OpenCV...
echo ============================================================
pip install mediapipe==0.10.14
pip install opencv-python==4.10.0.84
if errorlevel 1 (
    echo ERROR: Failed to install MediaPipe/OpenCV
    pause
    exit /b 1
)
echo MediaPipe and OpenCV installed successfully!
echo.

REM Install remaining dependencies
echo ============================================================
echo Step 7: Installing remaining dependencies...
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
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install pytest-cov==4.1.0
pip install requests==2.31.0
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo All dependencies installed successfully!
echo.

REM Verify installation
echo ============================================================
echo Step 8: Verifying installation...
echo ============================================================
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import sqlalchemy; print('SQLAlchemy:', sqlalchemy.__version__)"
echo.

echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Virtual environment created at: venv\
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.
echo Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run health check: python health_check.py
echo 3. Run tests: python run_tests.py
echo 4. Start server: uvicorn main:app --reload
echo.
pause
