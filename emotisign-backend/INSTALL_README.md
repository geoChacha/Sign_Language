# Installation Scripts

Quick reference for installation scripts.

## 🚀 Quick Start (One Command)

### Windows:
```cmd
quickstart.bat
```

This will:
1. ✅ Install all dependencies
2. ✅ Run health check
3. ✅ Run tests
4. ✅ Start the server

### Linux/Mac:
```bash
# Install dependencies
chmod +x install.sh
./install.sh

# Run health check
python3 health_check.py

# Run tests
python3 run_tests.py

# Start server
uvicorn main:app --reload
```

---

## 📦 Installation Only

### Windows:
```cmd
install.bat
```

### Linux/Mac:
```bash
chmod +x install.sh
./install.sh
```

This installs:
- ✅ PyTorch (CPU version)
- ✅ FastAPI and web dependencies
- ✅ SQLAlchemy and database dependencies
- ✅ MediaPipe and OpenCV
- ✅ Testing dependencies (pytest, requests)

---

## ✅ Verify Installation

```bash
python health_check.py
```

Expected output:
```
✓ All checks passed! ML service is ready.
```

---

## 🧪 Run Tests

```bash
python run_tests.py
```

Expected output:
```
✓ All tests passed!
```

---

## 🌐 Start Server

```bash
uvicorn main:app --reload
```

Server will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 📚 Full Documentation

For detailed setup instructions, see:
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete step-by-step guide
- **[TESTING.md](TESTING.md)** - Testing guide
- **[README.md](README.md)** - Project overview

---

## 🐛 Troubleshooting

### "Python not found"
- Windows: Reinstall Python and check "Add to PATH"
- Linux: `sudo apt install python3.10`

### "pip not found"
```bash
python -m ensurepip --upgrade
```

### "Module not found"
Run installation script again:
```bash
install.bat  # Windows
./install.sh # Linux/Mac
```

### "Port already in use"
```bash
uvicorn main:app --reload --port 8001
```

---

## 💡 What Gets Installed

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.1 | Deep learning framework |
| torchvision | 0.15.2 | Vision utilities |
| mediapipe | 0.9.3 | Keypoint extraction |
| opencv-python | 4.8.1.78 | Video processing |
| fastapi | 0.115.5 | Web framework |
| uvicorn | 0.32.1 | ASGI server |
| sqlalchemy | 2.0.36 | Database ORM |
| pytest | 7.4.3 | Testing framework |
| numpy | 1.24.3 | Array operations |

And more... (see `requirements.txt` for full list)

---

## 🎯 Next Steps

After installation:

1. ✅ Run health check: `python health_check.py`
2. ✅ Run tests: `python run_tests.py`
3. ✅ Start server: `uvicorn main:app --reload`
4. ✅ Test API: `curl http://localhost:8000/api/ml/health`
5. ✅ Open docs: http://localhost:8000/docs

---

## 🔧 Advanced Options

### Install with GPU support:
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Install specific package:
```bash
pip install sqlalchemy==2.0.36
```

### Update all packages:
```bash
pip install -r requirements.txt --upgrade
```

---

**Need help? See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.**
