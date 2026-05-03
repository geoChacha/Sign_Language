# 🚀 START HERE - EmotiSign Backend Setup

## Quick Start (3 Steps)

### Step 1: Install Dependencies

**Windows:**
```cmd
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

### Step 2: Verify Installation

```bash
python health_check.py
```

Should show: `✓ All checks passed!`

### Step 3: Start Server

```bash
uvicorn main:app --reload
```

Server running at: http://localhost:8000

---

## ✅ What You Need

- Python 3.9 or 3.10
- 8GB RAM minimum
- Internet connection (for installation)

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| **install.bat** / **install.sh** | Install all dependencies |
| **health_check.py** | Verify installation |
| **run_tests.py** | Run unit tests |
| **test_system.py** | Test complete system |
| **SETUP_GUIDE.md** | Detailed setup instructions |
| **TESTING.md** | Testing guide |

---

## 🎯 Installation Order

1. **Install Python 3.9/3.10** (if not installed)
2. **Run installation script** (`install.bat` or `install.sh`)
3. **Run health check** (`python health_check.py`)
4. **Run tests** (`python run_tests.py`)
5. **Start server** (`uvicorn main:app --reload`)

---

## 🧪 Testing

### Quick Tests:
```bash
python run_tests.py
```

### Full System Tests:
```bash
python test_system.py
```

### API Tests:
```bash
curl http://localhost:8000/api/ml/health
curl http://localhost:8000/api/ml/stats
```

---

## 📖 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup guide with troubleshooting
- **[INSTALL_README.md](INSTALL_README.md)** - Installation scripts reference
- **[TESTING.md](TESTING.md)** - Testing guide
- **[QUICK_TEST.md](QUICK_TEST.md)** - Quick testing reference
- **[README.md](README.md)** - Project overview
- **[docs/api.md](docs/api.md)** - API documentation
- **[docs/deployment.md](docs/deployment.md)** - Deployment guide
- **[docs/troubleshooting.md](docs/troubleshooting.md)** - Troubleshooting

---

## 🐛 Common Issues

### "Python not found"
Install Python 3.10 from https://www.python.org/

### "Module not found"
Run installation script again: `install.bat` or `./install.sh`

### "Model file not found"
Verify files exist:
```bash
ls app/ml/models/wlasl100/
# Should show: best_model.pth, vocab.json, temperature.json
```

### "Port already in use"
Use different port:
```bash
uvicorn main:app --reload --port 8001
```

---

## 🎉 Success Checklist

- [ ] Python 3.9/3.10 installed
- [ ] Dependencies installed (`install.bat` or `install.sh`)
- [ ] Health check passed (`python health_check.py`)
- [ ] Tests passed (`python run_tests.py`)
- [ ] Server started (`uvicorn main:app --reload`)
- [ ] API working (`curl http://localhost:8000/api/ml/health`)

---

## 🌐 API Endpoints

Once server is running:

- **Health**: http://localhost:8000/api/ml/health
- **Stats**: http://localhost:8000/api/ml/stats
- **Docs**: http://localhost:8000/docs
- **Sign-to-Text**: POST http://localhost:8000/api/ml/sign-to-text
- **Validation**: POST http://localhost:8000/api/ml/validate-environment

---

## 💡 Next Steps

1. ✅ **Test API endpoints** - Use curl or Postman
2. ✅ **Setup frontend** - See `emotisign-frontend/README.md`
3. ✅ **Test complete workflow** - Record ASL sign and translate
4. ✅ **Deploy to production** - See `docs/deployment.md`

---

## 🆘 Need Help?

1. Check **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for detailed instructions
2. Check **[docs/troubleshooting.md](docs/troubleshooting.md)** for solutions
3. Run `python health_check.py` to diagnose issues
4. Check server logs for errors

---

**Ready to start? Run `install.bat` (Windows) or `./install.sh` (Linux/Mac)!**
