# WLASL-100 ML Service Deployment Guide

This guide covers deploying the WLASL-100 ASL recognition service to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Model Files Deployment](#model-files-deployment)
4. [GPU Configuration](#gpu-configuration)
5. [Application Deployment](#application-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Minimum (CPU-only):**
- CPU: 4 cores, 2.5GHz+
- RAM: 8GB
- Storage: 10GB
- Network: 100Mbps

**Recommended (GPU-accelerated):**
- CPU: 8 cores, 3.0GHz+
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (e.g., GTX 1650, RTX 3060)
- Storage: 20GB SSD
- Network: 1Gbps

### Software Requirements

- **Operating System:** Ubuntu 20.04+ / Windows Server 2019+ / macOS 11+
- **Python:** 3.9 or 3.10 (3.11+ not tested)
- **CUDA:** 11.8 (for GPU acceleration)
- **cuDNN:** 8.x for CUDA 11.8
- **Docker:** 20.10+ (optional, for containerized deployment)

---

## Environment Setup

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y libgl1-mesa-glx libglib2.0-0  # For OpenCV
```

**Windows:**
- Install Python 3.10 from [python.org](https://www.python.org/downloads/)
- Install Visual C++ Redistributable (required for MediaPipe)

**macOS:**
```bash
brew install python@3.10
```

### 2. Create Virtual Environment

```bash
cd emotisign-backend
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

**For CPU-only deployment:**
```bash
pip install -r requirements.txt
```

**For GPU-accelerated deployment:**
```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install mediapipe==0.9.3 opencv-python==4.8.1.78 numpy==1.24.3
pip install fastapi uvicorn[standard] sqlalchemy aiosqlite python-jose[cryptography] passlib[bcrypt] python-multipart aiofiles
```

### 4. Configure Environment Variables

Create `.env` file in `emotisign-backend/`:

```env
# Application
SECRET_KEY=<generate-strong-random-key>
DATABASE_URL=sqlite+aiosqlite:///./emotisign.db
UPLOAD_DIR=uploads/videos

# ML Model Configuration
MODEL_PATH=app/ml/models/wlasl100/best_model.pth
VOCAB_PATH=app/ml/models/wlasl100/vocab.json
TEMPERATURE_PATH=app/ml/models/wlasl100/temperature.json
ML_DEVICE=cuda  # or 'cpu' for CPU-only

# Inference Configuration
USE_TTA=true
CONFIDENCE_THRESHOLD=0.25
MAX_VIDEO_SIZE_MB=50
MAX_VIDEO_DURATION_SEC=30
```

**Generate SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Model Files Deployment

### 1. Verify Model Files

Ensure these files exist in `app/ml/models/wlasl100/`:

```
app/ml/models/wlasl100/
├── best_model.pth        # 25MB
├── vocab.json            # 5KB
└── temperature.json      # 1KB
```

### 2. Model File Permissions

```bash
chmod 644 app/ml/models/wlasl100/*.pth
chmod 644 app/ml/models/wlasl100/*.json
```

### 3. Verify Model Loading

Run health check:
```bash
python health_check.py
```

Expected output:
```
✓ Found: app/ml/models/wlasl100/best_model.pth (25.00 MB)
✓ Found: app/ml/models/wlasl100/vocab.json
✓ PyTorch version: 2.0.1
✓ CUDA available: 11.8
✓ GPU: NVIDIA GeForce RTX 3070
✓ MediaPipe initialized successfully
✓ All checks passed
```

---

## GPU Configuration

### 1. Install NVIDIA Drivers

**Ubuntu:**
```bash
# Check current driver
nvidia-smi

# Install latest driver (if needed)
sudo apt install nvidia-driver-525
sudo reboot
```

**Windows:**
- Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

### 2. Install CUDA Toolkit 11.8

**Ubuntu:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**Windows:**
- Download from [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 3. Install cuDNN

1. Download cuDNN 8.x for CUDA 11.8 from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. Extract and copy files:

**Ubuntu:**
```bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 4. Verify GPU Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Application Deployment

### Option 1: Direct Deployment with Uvicorn

**Development:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 2: Deployment with Gunicorn + Uvicorn Workers

**Install Gunicorn:**
```bash
pip install gunicorn
```

**Run:**
```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Option 3: Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For GPU support, install PyTorch with CUDA
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t emotisign-backend .
docker run -p 8000:8000 --gpus all emotisign-backend
```

### Option 4: Systemd Service (Linux)

**Create service file:** `/etc/systemd/system/emotisign.service`

```ini
[Unit]
Description=EmotiSign Backend Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/emotisign-backend
Environment="PATH=/opt/emotisign-backend/venv/bin"
ExecStart=/opt/emotisign-backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable emotisign
sudo systemctl start emotisign
sudo systemctl status emotisign
```

---

## Monitoring and Logging

### 1. Application Logs

**View logs (systemd):**
```bash
sudo journalctl -u emotisign -f
```

**View logs (Docker):**
```bash
docker logs -f <container-id>
```

### 2. Performance Monitoring

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Monitor system resources:**
```bash
htop
```

### 3. Application Metrics

**Get ML service stats:**
```bash
curl http://localhost:8000/api/ml/stats
```

**Health check:**
```bash
curl http://localhost:8000/api/ml/health
```

### 4. Logging Configuration

Add to `main.py`:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotisign.log'),
        logging.StreamHandler()
    ]
)
```

### 5. Error Tracking (Optional)

**Sentry integration:**
```bash
pip install sentry-sdk[fastapi]
```

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)
```

---

## Performance Optimization

### 1. GPU Memory Optimization

**Reduce batch size:**
- TTA uses 4 variants per video
- Disable TTA for lower memory usage: `USE_TTA=false`

**Enable mixed precision:**
- Already enabled via `torch.cuda.amp.autocast()`

### 2. CPU Optimization

**Increase worker processes:**
```bash
uvicorn main:app --workers 8  # Match CPU core count
```

**Disable TTA:**
```env
USE_TTA=false
```

### 3. Caching

**Cache validation results:**
- Implement Redis caching for repeated validation frames
- Cache model predictions for identical videos

### 4. Load Balancing

**Nginx configuration:**
```nginx
upstream emotisign_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name api.emotisign.com;

    location /api/ml {
        proxy_pass http://emotisign_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
        client_max_body_size 50M;
    }
}
```

### 5. Database Optimization

**Use PostgreSQL for production:**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/emotisign
```

---

## Troubleshooting

### GPU Not Detected

**Check NVIDIA driver:**
```bash
nvidia-smi
```

**Check CUDA:**
```bash
nvcc --version
```

**Check PyTorch CUDA:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution:**
- Reinstall NVIDIA drivers
- Reinstall CUDA toolkit
- Reinstall PyTorch with correct CUDA version

### Out of Memory Errors

**Symptoms:**
- "CUDA out of memory" errors
- Application crashes during inference

**Solutions:**
1. Disable TTA: `USE_TTA=false`
2. Reduce video resolution
3. Limit concurrent requests
4. Add GPU memory monitoring

**Automatic fallback:**
- System automatically falls back to CPU on GPU OOM

### Slow Inference

**CPU inference is slow:**
- Enable GPU acceleration
- Disable TTA
- Reduce video length

**GPU inference is slow:**
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is properly installed
- Check for thermal throttling

### MediaPipe Errors

**"Failed to initialize MediaPipe":**
- Ensure mediapipe==0.9.3 (not 0.10+)
- Install Visual C++ Redistributable (Windows)
- Check OpenCV installation

### Model Loading Errors

**"Model checkpoint not found":**
- Verify `MODEL_PATH` in `.env`
- Check file permissions
- Ensure model files are copied correctly

**"Vocabulary file not found":**
- Verify `VOCAB_PATH` in `.env`
- Check file exists and is readable

---

## Security Checklist

- [ ] Change `SECRET_KEY` to strong random value
- [ ] Restrict CORS `allow_origins` to frontend domain
- [ ] Use HTTPS/TLS (Nginx/Traefik)
- [ ] Implement rate limiting (slowapi)
- [ ] Validate file content (not just MIME type)
- [ ] Set up firewall rules
- [ ] Enable application logging
- [ ] Set up monitoring and alerts
- [ ] Regular security updates
- [ ] Backup model files and database

---

## Production Checklist

- [ ] GPU drivers and CUDA installed
- [ ] Model files deployed and verified
- [ ] Environment variables configured
- [ ] Health check passing
- [ ] Performance benchmarks met
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backup strategy in place
- [ ] Load balancing configured (if needed)
- [ ] SSL/TLS certificates installed
- [ ] CORS configured for production domain
- [ ] Rate limiting enabled
- [ ] Error tracking configured (Sentry)
- [ ] Documentation updated

---

## Support

For deployment issues:
1. Run `python health_check.py` to diagnose
2. Check application logs
3. Review this guide's troubleshooting section
4. Contact the development team

---

## Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [PyTorch Production](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/)
