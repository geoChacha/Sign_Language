# 🐍 Virtual Environment Setup Guide

This guide will help you create a fresh Python virtual environment with all required dependencies.

---

## Why Use a Virtual Environment?

A virtual environment:
- ✅ Isolates project dependencies from system Python
- ✅ Prevents version conflicts between projects
- ✅ Ensures consistent package versions
- ✅ Makes deployment easier

---

## Quick Start

### Windows

```cmd
cd emotisign-backend
setup_venv.bat
```

### Linux/Mac

```bash
cd emotisign-backend
chmod +x setup_venv.sh
./setup_venv.sh
```

The script will:
1. Create a new virtual environment in `venv/`
2. Install NumPy first (critical for compatibility)
3. Install PyTorch (CPU version)
4. Install MediaPipe and OpenCV (compatible versions)
5. Install all other dependencies
6. Verify the installation

**Time required:** 3-5 minutes

---

## Manual Setup (Alternative)

If the automated script doesn't work, follow these steps:

### 1. Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install Dependencies (IN ORDER)

**IMPORTANT:** Install in this exact order to avoid compatibility issues!

```bash
# Step 1: NumPy (foundation package)
pip install numpy==1.26.4

# Step 2: PyTorch (CPU version)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Step 3: MediaPipe and OpenCV (compatible versions)
pip install mediapipe==0.10.14
pip install opencv-python==4.10.0.84

# Step 4: All other dependencies
pip install -r requirements.txt
```

---

## Activating the Virtual Environment

You need to activate the virtual environment every time you open a new terminal.

### Windows

```cmd
cd emotisign-backend
venv\Scripts\activate.bat
```

You should see `(venv)` at the beginning of your command prompt.

### Linux/Mac

```bash
cd emotisign-backend
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

---

## Deactivating the Virtual Environment

When you're done working:

```bash
deactivate
```

---

## Verifying Installation

After setup, verify everything is installed correctly:

```bash
# Activate virtual environment first!
python health_check.py
```

Expected output:
```
✓ Model files check passed
✓ GPU check passed (or CPU fallback)
✓ MediaPipe check passed
✓ All checks passed!
```

---

## Running the Server

With the virtual environment activated:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
✅ WLASL-100 model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Troubleshooting

### Issue: "python: command not found"

**Solution:** Install Python 3.9 or 3.10 from https://www.python.org/

### Issue: "No module named 'numpy._core'"

**Solution:** This means numpy/mediapipe/opencv versions are incompatible. Run the setup script again to install compatible versions.

### Issue: "Permission denied" (Linux/Mac)

**Solution:** Make the script executable:
```bash
chmod +x setup_venv.sh
```

### Issue: Virtual environment not activating

**Windows:**
```cmd
# Try PowerShell activation instead
venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
# Make sure you use 'source'
source venv/bin/activate
```

### Issue: Packages already installed but wrong versions

**Solution:** Delete the virtual environment and start fresh:

**Windows:**
```cmd
rmdir /s /q venv
setup_venv.bat
```

**Linux/Mac:**
```bash
rm -rf venv
./setup_venv.sh
```

---

## GPU Support (Optional)

The default installation uses CPU-only PyTorch. For GPU acceleration:

### 1. Check CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version" in the output.

### 2. Install PyTorch with CUDA

**For CUDA 11.8:**
```bash
pip uninstall torch torchvision -y
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip uninstall torch torchvision -y
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Update .env

```env
ML_DEVICE=cuda
```

### 4. Verify GPU

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Package Versions

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Array operations (MUST be this version) |
| torch | 2.0.1 | Deep learning framework |
| mediapipe | 0.10.14 | Keypoint extraction (compatible with numpy 1.26) |
| opencv-python | 4.10.0.84 | Video processing (compatible with numpy 1.26) |
| fastapi | 0.115.5 | Web framework |
| sqlalchemy | 2.0.36 | Database ORM |

**Note:** The specific versions are critical for compatibility. Don't upgrade without testing!

---

## Next Steps

After successful setup:

1. ✅ **Verify installation:** `python health_check.py`
2. ✅ **Run tests:** `python run_tests.py`
3. ✅ **Start server:** `uvicorn main:app --reload`
4. ✅ **Test API:** `curl http://localhost:8000/api/ml/health`

---

## Additional Resources

- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **MediaPipe Documentation:** https://google.github.io/mediapipe/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/

---

**Need help?** Check the troubleshooting section or see `docs/troubleshooting.md`
