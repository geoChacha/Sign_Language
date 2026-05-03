# WLASL-100 ML Service Troubleshooting Guide

This guide helps diagnose and resolve common issues with the WLASL-100 ASL recognition service.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Model Loading Issues](#model-loading-issues)
4. [GPU and CUDA Issues](#gpu-and-cuda-issues)
5. [Inference Issues](#inference-issues)
6. [Performance Issues](#performance-issues)
7. [API Issues](#api-issues)
8. [Frontend Integration Issues](#frontend-integration-issues)

---

## Quick Diagnostics

### Run Health Check

First, run the health check script to identify issues:

```bash
cd emotisign-backend
python health_check.py
```

This will check:
- Model files existence
- PyTorch installation
- CUDA availability
- MediaPipe initialization
- OpenCV installation

### Check Logs

**View application logs:**
```bash
# If running with uvicorn
tail -f emotisign.log

# If running with systemd
sudo journalctl -u emotisign -f

# If running with Docker
docker logs -f <container-id>
```

### Test API Endpoints

**Test health endpoint:**
```bash
curl http://localhost:8000/api/ml/health
```

**Test stats endpoint:**
```bash
curl http://localhost:8000/api/ml/stats
```

---

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Cause:** PyTorch not installed

**Solution:**
```bash
# For CPU-only
pip install torch==2.0.1 torchvision==0.15.2

# For GPU (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "ModuleNotFoundError: No module named 'mediapipe'"

**Cause:** MediaPipe not installed

**Solution:**
```bash
pip install mediapipe==0.9.3
```

**Note:** Must use version 0.9.3. Newer versions (0.10+) have breaking API changes.

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Cause:** OpenCV not installed

**Solution:**
```bash
pip install opencv-python==4.8.1.78
```

### Issue: "ImportError: DLL load failed" (Windows)

**Cause:** Missing Visual C++ Redistributable

**Solution:**
1. Download and install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Restart terminal
3. Reinstall mediapipe: `pip install --force-reinstall mediapipe==0.9.3`

### Issue: "libGL.so.1: cannot open shared object file" (Linux)

**Cause:** Missing OpenGL libraries

**Solution:**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

---

## Model Loading Issues

### Issue: "FileNotFoundError: Model checkpoint not found"

**Cause:** Model file missing or incorrect path

**Diagnosis:**
```bash
ls -lh app/ml/models/wlasl100/best_model.pth
```

**Solution:**
1. Verify model file exists
2. Check `MODEL_PATH` in `.env`:
   ```env
   MODEL_PATH=app/ml/models/wlasl100/best_model.pth
   ```
3. Ensure file permissions are correct:
   ```bash
   chmod 644 app/ml/models/wlasl100/best_model.pth
   ```

### Issue: "FileNotFoundError: Vocabulary file not found"

**Cause:** Vocabulary file missing or incorrect path

**Solution:**
1. Verify vocab file exists:
   ```bash
   ls -lh app/ml/models/wlasl100/vocab.json
   ```
2. Check `VOCAB_PATH` in `.env`:
   ```env
   VOCAB_PATH=app/ml/models/wlasl100/vocab.json
   ```

### Issue: "RuntimeError: Error(s) in loading state_dict"

**Cause:** Model architecture mismatch with checkpoint

**Diagnosis:**
```python
import torch
checkpoint = torch.load('app/ml/models/wlasl100/best_model.pth', map_location='cpu')
print(checkpoint.keys())
```

**Solution:**
1. Verify model architecture in `app/ml/wlasl_model.py` matches checkpoint
2. Check model parameters:
   - `feature_dim=126`
   - `num_classes=100`
   - `d_model=192`
   - `num_layers=2`
3. Ensure checkpoint contains `model_state_dict` key

### Issue: "Model loads but predictions are random"

**Cause:** Model not in evaluation mode or weights not loaded correctly

**Solution:**
1. Verify model is set to eval mode:
   ```python
   model.eval()
   ```
2. Check model weights loaded:
   ```python
   print(model.state_dict().keys())
   ```
3. Verify temperature scaling is applied

---

## GPU and CUDA Issues

### Issue: "CUDA not available" (torch.cuda.is_available() returns False)

**Diagnosis:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Solution:**

**If nvidia-smi fails:**
1. Install/update NVIDIA drivers:
   ```bash
   # Ubuntu
   sudo apt install nvidia-driver-525
   sudo reboot
   ```

**If CUDA version mismatch:**
1. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

**If still not working:**
1. Check CUDA installation:
   ```bash
   ls /usr/local/cuda/
   ```
2. Add to PATH (if needed):
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Check VRAM requirements
# - Model: ~500MB
# - Inference (TTA): ~1.5GB
# - Total: ~2GB minimum
```

**Solutions:**

**1. Disable TTA (reduces memory by 4x):**
```env
USE_TTA=false
```

**2. Reduce video resolution:**
- Resize videos to 640x480 before processing

**3. Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

**4. Automatic CPU fallback:**
- System automatically falls back to CPU on OOM
- Check logs for "Falling back to CPU" message

**5. Increase GPU memory:**
- Close other GPU applications
- Use GPU with more VRAM

### Issue: "GPU utilization is low"

**Diagnosis:**
```bash
watch -n 1 nvidia-smi
```

**Causes and Solutions:**

**1. CPU bottleneck:**
- Keypoint extraction is CPU-bound
- Solution: Use faster CPU or reduce video resolution

**2. I/O bottleneck:**
- Video loading is slow
- Solution: Use SSD storage, reduce video size

**3. Single-threaded inference:**
- Only one video processed at a time
- Solution: Increase worker processes

---

## Inference Issues

### Issue: "Unable to process video file"

**Cause:** Video format not supported or corrupted

**Diagnosis:**
```bash
# Check video with ffmpeg
ffmpeg -i video.mp4

# Check video with OpenCV
python -c "import cv2; cap = cv2.VideoCapture('video.mp4'); print(f'Opened: {cap.isOpened()}, Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')"
```

**Solutions:**

**1. Convert video format:**
```bash
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

**2. Supported formats:**
- MP4 (recommended)
- AVI
- MOV
- WEBM

**3. Check video properties:**
- Duration: < 30 seconds
- Size: < 50MB
- Resolution: 640x480 or higher
- FPS: 15-30

### Issue: "MediaPipe extraction failed"

**Symptoms:**
- Zero-filled keypoints
- Low confidence predictions
- "No hands detected" warnings

**Diagnosis:**
```python
from app.ml.keypoint_extractor import KeypointExtractor
import cv2

extractor = KeypointExtractor()
frame = cv2.imread('test_frame.jpg')
keypoints = extractor.extract_keypoints_from_frame(frame)
print(f"Non-zero keypoints: {(keypoints != 0).sum()}")
```

**Solutions:**

**1. Improve video quality:**
- Ensure good lighting
- Clear background
- Hands visible and in frame
- Camera stable (not shaky)

**2. Check MediaPipe version:**
```bash
pip show mediapipe
# Should be 0.9.3
```

**3. Adjust MediaPipe settings:**
- Increase `model_complexity` (already at 2)
- Adjust `min_detection_confidence`

### Issue: "Predictions are always the same"

**Cause:** Model stuck or not processing input correctly

**Diagnosis:**
```python
# Check if model is in eval mode
print(model.training)  # Should be False

# Check input shape
print(input_tensor.shape)  # Should be (batch_size, 64, 126)

# Check output logits
print(logits)  # Should vary for different inputs
```

**Solutions:**

**1. Verify model is loaded:**
```bash
curl http://localhost:8000/api/ml/stats
# Check "model_loaded": true
```

**2. Check input preprocessing:**
- Keypoints normalized to [0, 1]
- Sequence resampled to 64 frames
- Correct feature order (left hand, right hand)

**3. Verify temperature scaling:**
- Check `temperature.json` exists
- Temperature should be ~1.0

### Issue: "Low confidence scores"

**Symptoms:**
- All predictions < 0.3 confidence
- Top-5 predictions very similar

**Causes and Solutions:**

**1. Poor video quality:**
- Improve lighting
- Use clear background
- Ensure hands are visible

**2. Out-of-vocabulary signs:**
- Model only knows 100 ASL signs
- Check if sign is in vocabulary:
  ```bash
  cat app/ml/models/wlasl100/vocab.json
  ```

**3. Incorrect signing:**
- Verify sign is performed correctly
- Check sign duration (2-5 seconds recommended)

**4. Model calibration:**
- Temperature scaling may need adjustment
- Check `temperature.json` value

---

## Performance Issues

### Issue: "Inference is very slow"

**Diagnosis:**
```bash
# Check device
curl http://localhost:8000/api/ml/stats
# Look at "device" field

# Check processing time
# Should be:
# - GPU with TTA: 200-500ms
# - GPU without TTA: 50-125ms
# - CPU with TTA: 1-2s
# - CPU without TTA: 250-500ms
```

**Solutions:**

**1. Enable GPU acceleration:**
```env
ML_DEVICE=cuda
```

**2. Disable TTA:**
```env
USE_TTA=false
```

**3. Reduce video length:**
- Trim videos to 5-10 seconds
- Shorter videos process faster

**4. Optimize video resolution:**
- Use 640x480 resolution
- Higher resolutions don't improve accuracy

**5. Increase worker processes:**
```bash
uvicorn main:app --workers 4
```

### Issue: "High memory usage"

**Diagnosis:**
```bash
# Check memory usage
htop

# Check GPU memory
nvidia-smi
```

**Solutions:**

**1. Disable TTA:**
```env
USE_TTA=false
```

**2. Limit concurrent requests:**
- Use request queue
- Implement rate limiting

**3. Clear cache periodically:**
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Issue: "Server crashes under load"

**Symptoms:**
- Server stops responding
- Out of memory errors
- Connection timeouts

**Solutions:**

**1. Increase timeout:**
```bash
uvicorn main:app --timeout-keep-alive 120
```

**2. Add request queue:**
- Implement async queue for video processing
- Limit concurrent inference requests

**3. Add health checks:**
- Monitor memory usage
- Restart on high memory

**4. Use load balancer:**
- Distribute requests across multiple instances
- See deployment guide for Nginx configuration

---

## API Issues

### Issue: "CORS errors in browser"

**Symptoms:**
```
Access to fetch at 'http://localhost:8000/api/ml/...' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Solution:**

Update CORS configuration in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: "413 Request Entity Too Large"

**Cause:** Video file exceeds size limit

**Solution:**

**1. Increase limit in `.env`:**
```env
MAX_VIDEO_SIZE_MB=100
```

**2. Configure Nginx (if using):**
```nginx
client_max_body_size 100M;
```

**3. Compress video:**
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 28 output.mp4
```

### Issue: "504 Gateway Timeout"

**Cause:** Processing takes too long

**Solution:**

**1. Increase timeout:**
```nginx
proxy_read_timeout 300s;
```

**2. Optimize inference:**
- Disable TTA
- Use GPU
- Reduce video length

### Issue: "Connection refused"

**Diagnosis:**
```bash
# Check if server is running
curl http://localhost:8000/api/ml/health

# Check port
netstat -tulpn | grep 8000
```

**Solutions:**

**1. Start server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**2. Check firewall:**
```bash
sudo ufw allow 8000
```

**3. Check binding:**
- Ensure server binds to `0.0.0.0` not `127.0.0.1`

---

## Frontend Integration Issues

### Issue: "Validation not working"

**Symptoms:**
- Validation status always "not_ready"
- No validation feedback

**Diagnosis:**
```bash
# Test validation endpoint
curl -X POST http://localhost:8000/api/ml/validate-environment \
  -F "frame=@test_frame.jpg"
```

**Solutions:**

**1. Check frame capture:**
```javascript
// Ensure video is playing
console.log(video.readyState);  // Should be 4

// Check canvas size
console.log(canvas.width, canvas.height);  // Should match video
```

**2. Check validation interval:**
```javascript
// Should be 200ms (5 fps)
setInterval(sendFrameForValidation, 200);
```

**3. Check API URL:**
```javascript
const API_URL = process.env.NEXT_PUBLIC_API_URL;
console.log(API_URL);  // Should be http://localhost:8000
```

### Issue: "Recording not starting"

**Symptoms:**
- "Start Recording" button disabled
- Validation stuck at "validating"

**Solutions:**

**1. Check validation state:**
```javascript
console.log(validationState.overall);  // Should be "ready"
```

**2. Check MediaRecorder support:**
```javascript
console.log(MediaRecorder.isTypeSupported('video/webm'));  // Should be true
```

**3. Check webcam permissions:**
```javascript
navigator.permissions.query({ name: 'camera' })
  .then(result => console.log(result.state));  // Should be "granted"
```

### Issue: "Upload fails"

**Symptoms:**
- "Failed to process video" error
- 400 or 500 status code

**Diagnosis:**
```javascript
// Check blob size
console.log(blob.size);  // Should be < 50MB

// Check blob type
console.log(blob.type);  // Should be video/webm
```

**Solutions:**

**1. Check file size:**
```javascript
if (blob.size > 50 * 1024 * 1024) {
  console.error('File too large');
}
```

**2. Check video format:**
```javascript
// Try different codec
const mediaRecorder = new MediaRecorder(stream, {
  mimeType: 'video/webm;codecs=vp8'
});
```

**3. Check API response:**
```javascript
const response = await fetch(url, { method: 'POST', body: formData });
console.log(response.status, await response.text());
```

---

## Getting Help

If you're still experiencing issues:

1. **Run health check:**
   ```bash
   python health_check.py
   ```

2. **Check logs:**
   ```bash
   tail -f emotisign.log
   ```

3. **Test API endpoints:**
   ```bash
   curl http://localhost:8000/api/ml/health
   curl http://localhost:8000/api/ml/stats
   ```

4. **Gather information:**
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA version: `nvcc --version`
   - GPU info: `nvidia-smi`
   - OS: `uname -a` (Linux) or `ver` (Windows)

5. **Contact support:**
   - Include health check output
   - Include relevant log excerpts
   - Describe steps to reproduce
   - Include error messages

---

## Additional Resources

- [README.md](../README.md) - Setup instructions
- [docs/api.md](api.md) - API documentation
- [docs/deployment.md](deployment.md) - Deployment guide
- [PyTorch Troubleshooting](https://pytorch.org/docs/stable/notes/faq.html)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
