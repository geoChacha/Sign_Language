# EmotiSign Backend - Complete Setup Guide

This guide will walk you through setting up and running the EmotiSign backend with WLASL-100 model integration.

## Prerequisites

- **Python 3.9 or 3.10** (3.11+ not tested)
- **pip** (Python package manager)
- **Git** (to clone the repository)
- **8GB RAM minimum** (16GB recommended)
- **GPU optional** (NVIDIA GPU with CUDA 11.8 for faster inference)

---

## Step 1: Install Python

### Windows:
1. Download Python 3.10 from https://www.python.org/downloads/
2. Run installer
3. ✅ **Check "Add Python to PATH"**
4. Click "Install Now"
5. Verify installation:
   ```cmd
   python --version
   ```

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3.10 python3-pip
python3 --version
```

### macOS:
```bash
brew install python@3.10
python3 --version
```

---

## Step 2: Navigate to Backend Directory

```bash
cd emotisign-backend
```

---

## Step 3: Install All Dependencies

### Option A: Automatic Installation (Recommended)

**Windows:**
```cmd
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

This will:
- ✅ Upgrade pip
- ✅ Install PyTorch (CPU version)
- ✅ Install FastAPI and all web dependencies
- ✅ Install SQLAlchemy and database dependencies
- ✅ Install MediaPipe and OpenCV
- ✅ Install testing dependencies
- ✅ Verify all installations

### Option B: Manual Installation

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

**Note:** For GPU support, see [GPU Setup](#gpu-setup-optional) below.

---

## Step 4: Verify Installation

Run the health check script:

```bash
python health_check.py
```

Expected output:
```
╔==========================================================╗
║          EmotiSign ML Service Health Check               ║
╚==========================================================╝

============================================================
Checking Model Files...
============================================================
✓ Found: app/ml/models/wlasl100/best_model.pth (25.00 MB)
✓ Found: app/ml/models/wlasl100/vocab.json

============================================================
Checking PyTorch...
============================================================
✓ PyTorch version: 2.0.1
⚠ No GPU available, will use CPU

============================================================
Checking MediaPipe...
============================================================
✓ MediaPipe version: 0.9.3
✓ MediaPipe Holistic initialized successfully

============================================================
Checking OpenCV...
============================================================
✓ OpenCV version: 4.8.1

============================================================
Checking NumPy...
============================================================
✓ NumPy version: 1.24.3

============================================================
Summary
============================================================
Model Files              ✓ PASS
PyTorch                  ✓ PASS
MediaPipe                ✓ PASS
OpenCV                   ✓ PASS
NumPy                    ✓ PASS

============================================================
✓ All checks passed! ML service is ready.
============================================================
```

---

## Step 5: Run Tests

### Quick Unit Tests:

```bash
python run_tests.py
```

This tests:
- ✅ Model architecture
- ✅ Sequence resampling
- ✅ TTA variants
- ✅ Statistics tracking

Expected output:
```
╔==========================================================╗
║               WLASL-100 Unit Tests                       ║
╚==========================================================╝

============================================================
Testing Model Architecture
============================================================

1. Testing model initialization...
   ✓ Model initialized
2. Testing forward pass...
   ✓ Forward pass successful, output shape: (1, 100)
3. Testing batch processing...
   ✓ Batch processing successful, output shape: (4, 100)
4. Testing parameter count...
   Total parameters: 412,356
   ✓ Parameter count is within expected range
5. Testing dropout behavior...
   ✓ Dropout is disabled in eval mode

✓ All model architecture tests passed!

============================================================
Testing ML Service
============================================================

1. Testing sequence resampling (upsampling)...
   ✓ Upsampling: (30, 126) → (64, 126)
2. Testing sequence resampling (downsampling)...
   ✓ Downsampling: (100, 126) → (64, 126)
3. Testing sequence resampling (padding)...
   ✓ Padding: (20, 126) → (64, 126)
4. Testing TTA variant generation...
   ✓ Generated 4 TTA variants
5. Testing statistics...
   ✓ Statistics: {...}

✓ All ML service tests passed!

============================================================
Summary
============================================================
Model Architecture             ✓ PASS
ML Service                     ✓ PASS

============================================================
✓ All tests passed!
============================================================
```

### Full System Tests (Optional):

```bash
python test_system.py
```

This tests:
- ✅ Model loading
- ✅ Keypoint extraction
- ✅ Inference
- ✅ Environment validation
- ✅ API endpoints

---

## Step 6: Start the Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The server is now running! 🎉

---

## Step 7: Test API Endpoints

Open a new terminal and test the endpoints:

### Health Check:
```bash
curl http://localhost:8000/api/ml/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Stats:
```bash
curl http://localhost:8000/api/ml/stats
```

Expected response:
```json
{
  "total_predictions": 0,
  "average_inference_time_ms": 0.0,
  "average_confidence": 0.0,
  "model_loaded": true,
  "device": "cpu"
}
```

### API Documentation:
Open in browser: http://localhost:8000/docs

---

## Step 8: Setup Frontend (Optional)

In a new terminal:

```bash
cd ../emotisign-frontend
npm install
npm run dev
```

Then open: http://localhost:3000/translate/sign-to-text

---

## Step 9: Test Complete Workflow

1. **Open frontend**: http://localhost:3000/translate/sign-to-text
2. **Click "Start Validation"**
3. **Grant webcam permissions**
4. **Wait for validation** (background, body, distance checks)
5. **Click "Start Recording"** when ready
6. **Perform an ASL sign** (2-5 seconds)
7. **Click "Stop Recording"**
8. **Wait for processing** (~1-2 seconds on CPU)
9. **View results** (recognized text, confidence, top 5 predictions)

---

## GPU Setup (Optional)

For faster inference (5-10x speedup), install GPU-enabled PyTorch:

### Prerequisites:
- NVIDIA GPU with CUDA support
- CUDA 11.8 installed
- cuDNN 8.x for CUDA 11.8

### Install GPU PyTorch:

**Windows/Linux:**
```bash
pip uninstall torch torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Update .env:
```env
ML_DEVICE=cuda
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution:**
- Windows: Reinstall Python and check "Add to PATH"
- Linux: `sudo apt install python3.10`
- Mac: `brew install python@3.10`

### Issue: "pip not found"

**Solution:**
```bash
python -m ensurepip --upgrade
```

### Issue: "ModuleNotFoundError: No module named 'sqlalchemy'"

**Solution:**
```bash
pip install sqlalchemy==2.0.36
```

Or run the installation script again:
```bash
install.bat  # Windows
./install.sh # Linux/Mac
```

### Issue: "Model file not found"

**Solution:**
Verify model files exist:
```bash
ls app/ml/models/wlasl100/
# Should show: best_model.pth, vocab.json, temperature.json
```

### Issue: "MediaPipe initialization failed"

**Solution:**
```bash
pip install mediapipe==0.9.3
```

Must use version 0.9.3 (not 0.10+).

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Use a different port
uvicorn main:app --reload --port 8001
```

### Issue: "CUDA out of memory"

**Solution:**
1. Disable TTA in `.env`:
   ```env
   USE_TTA=false
   ```
2. Or use CPU:
   ```env
   ML_DEVICE=cpu
   ```

---

## Configuration

Edit `.env` file to configure:

```env
# ML Model Configuration
MODEL_PATH=app/ml/models/wlasl100/best_model.pth
VOCAB_PATH=app/ml/models/wlasl100/vocab.json
TEMPERATURE_PATH=app/ml/models/wlasl100/temperature.json

# Device: 'cuda' for GPU, 'cpu' for CPU
ML_DEVICE=cpu

# Enable Test-Time Augmentation (improves accuracy but slower)
USE_TTA=true

# Confidence threshold
CONFIDENCE_THRESHOLD=0.25

# Video limits
MAX_VIDEO_SIZE_MB=50
MAX_VIDEO_DURATION_SEC=30
```

---

## Performance Expectations

### CPU (Intel i7)
- Model loading: 3-5 seconds
- Inference with TTA: 1-2 seconds per video
- Inference without TTA: 250-500ms per video

### GPU (NVIDIA RTX 3070)
- Model loading: 2-3 seconds
- Inference with TTA: 200-500ms per video
- Inference without TTA: 50-125ms per video

---

## Quick Reference

### Start Server:
```bash
uvicorn main:app --reload
```

### Run Tests:
```bash
python run_tests.py
```

### Health Check:
```bash
python health_check.py
```

### API Docs:
http://localhost:8000/docs

### Frontend:
http://localhost:3000/translate/sign-to-text

---

## Next Steps

1. ✅ **Test with real ASL signs** - Record videos and verify predictions
2. ✅ **Performance tuning** - Enable GPU, adjust TTA settings
3. ✅ **Deploy to production** - See `docs/deployment.md`
4. ✅ **Monitor and optimize** - Track inference times and accuracy

---

## Getting Help

- **API Documentation**: `docs/api.md`
- **Deployment Guide**: `docs/deployment.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **Testing Guide**: `TESTING.md`

---

## Summary Checklist

- [ ] Python 3.9/3.10 installed
- [ ] Dependencies installed (`install.bat` or `install.sh`)
- [ ] Health check passed (`python health_check.py`)
- [ ] Tests passed (`python run_tests.py`)
- [ ] Server started (`uvicorn main:app --reload`)
- [ ] API endpoints working (`curl http://localhost:8000/api/ml/health`)
- [ ] Frontend running (optional)
- [ ] Complete workflow tested

---

**Congratulations! Your EmotiSign backend is ready! 🎉**
