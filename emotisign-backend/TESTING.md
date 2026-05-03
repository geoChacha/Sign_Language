# Testing Guide for WLASL-100 Integration

This guide helps you test the WLASL-100 ASL recognition system.

## Quick Start

### 1. Install Dependencies

```bash
cd emotisign-backend
pip install -r requirements.txt
```

### 2. Run Simple Unit Tests (Recommended First)

This runs tests without requiring all app dependencies:

```bash
python run_tests.py
```

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

### 3. Run System Verification

This tests the complete system (requires all dependencies):

```bash
python test_system.py
```

Expected output:
```
╔==========================================================╗
║          WLASL-100 System Verification                   ║
╚==========================================================╝

============================================================
  Test 1: Model Loading
============================================================
✓ Model files found
  Using device: cuda
✓ Model loaded successfully
✓ Model is in evaluation mode
✓ Vocabulary loaded: 100 words

============================================================
  Test 2: Keypoint Extraction
============================================================
✓ KeypointExtractor initialized
✓ Keypoints extracted: shape (126,)
✓ Keypoints dtype is correct

============================================================
  Test 3: Inference with Dummy Data
============================================================
  Created dummy keypoints: shape (100, 126)
✓ Result has all required keys
  Recognized text: hello
  Confidence: 0.856
  Top 5: ['hello', 'help', 'home', 'have', 'hearing']
  Processing time: 342ms
✓ Confidence is in valid range
✓ Top 5 predictions returned

============================================================
  Test 4: Environment Validation
============================================================
✓ EnvironmentValidator initialized
✓ Result has all required keys
  Background: cluttered
  Body: not_visible
  Distance: out_of_range
  Overall: not_ready
✓ Overall status is valid

============================================================
  Test 5: API Endpoints
============================================================
✓ Health endpoint is accessible
  Status: healthy
  Model loaded: True
  Device: cuda
✓ Stats endpoint is accessible
  Total predictions: 0

============================================================
  Summary
============================================================
Model Loading                  ✓ PASS
Keypoint Extraction            ✓ PASS
Inference                      ✓ PASS
Environment Validation         ✓ PASS
API Endpoints                  ✓ PASS

============================================================
✓ All tests passed! System is working correctly.
============================================================
```

### 3. Run Unit Tests

```bash
pytest
```

Or with verbose output:

```bash
pytest -v
```

### 4. Test API Endpoints Manually

Start the server:

```bash
uvicorn main:app --reload
```

Then test endpoints:

```bash
# Health check
curl http://localhost:8000/api/ml/health

# Stats
curl http://localhost:8000/api/ml/stats

# Sign-to-text (with video file)
curl -X POST http://localhost:8000/api/ml/sign-to-text \
  -F "video=@test_video.mp4" \
  -F "use_tta=true"

# Environment validation (with image file)
curl -X POST http://localhost:8000/api/ml/validate-environment \
  -F "frame=@test_frame.jpg"
```

## Testing Checklist

### Backend Tests

- [ ] Run `python health_check.py` - All checks pass
- [ ] Run `python test_system.py` - All tests pass
- [ ] Run `pytest` - All unit tests pass
- [ ] Start server with `uvicorn main:app --reload`
- [ ] Test `/api/ml/health` endpoint
- [ ] Test `/api/ml/stats` endpoint
- [ ] Test `/api/ml/sign-to-text` with video file
- [ ] Test `/api/ml/validate-environment` with image file

### Frontend Tests

- [ ] Start frontend with `npm run dev`
- [ ] Navigate to `/translate/sign-to-text`
- [ ] Click "Start Validation"
- [ ] Grant webcam permissions
- [ ] Wait for validation to show "ready"
- [ ] Click "Start Recording"
- [ ] Perform ASL sign
- [ ] Click "Stop Recording"
- [ ] Wait for processing
- [ ] View results

### Integration Tests

- [ ] Backend and frontend running simultaneously
- [ ] Validation feedback updates in real-time
- [ ] Recording works without errors
- [ ] Upload completes successfully
- [ ] Results display correctly
- [ ] "Record Again" button works

## Common Issues

### Issue: "Model file not found"

**Solution:**
```bash
ls app/ml/models/wlasl100/
# Should show: best_model.pth, vocab.json, temperature.json
```

If files are missing, check the model setup in README.md.

### Issue: "CUDA not available"

**Solution:**
- For testing, CPU is fine (just slower)
- For production, install CUDA 11.8 and GPU-enabled PyTorch
- See README.md for GPU setup instructions

### Issue: "MediaPipe initialization failed"

**Solution:**
```bash
pip install mediapipe==0.9.3
```

Must use version 0.9.3 (not 0.10+).

### Issue: "Server not running" (test_system.py)

**Solution:**
```bash
# Start server in another terminal
uvicorn main:app --reload
```

Then run `python test_system.py` again.

### Issue: "Webcam not accessible" (frontend)

**Solution:**
- Grant camera permissions in browser
- Check if another app is using the webcam
- Try a different browser (Chrome recommended)

## Performance Expectations

### GPU (NVIDIA RTX 3070)
- Model loading: ~2-3 seconds
- Inference (with TTA): 200-500ms per video
- Inference (without TTA): 50-125ms per video
- Keypoint extraction: ~30-60ms per frame

### CPU (Intel i7)
- Model loading: ~3-5 seconds
- Inference (with TTA): 1-2s per video
- Inference (without TTA): 250-500ms per video
- Keypoint extraction: ~50-100ms per frame

## Next Steps

After all tests pass:

1. **Manual Testing**: Test with real ASL sign videos
2. **User Acceptance**: Have users test the system
3. **Performance Tuning**: Optimize based on real-world usage
4. **Deployment**: Deploy to staging/production environment

## Getting Help

If tests fail:

1. Check the error messages carefully
2. Review the troubleshooting section above
3. Check `docs/troubleshooting.md` for detailed solutions
4. Run `python health_check.py` to diagnose issues
5. Check application logs for errors

## Resources

- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Test README](tests/README.md)
