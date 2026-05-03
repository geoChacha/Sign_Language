# WLASL-100 ML API Documentation

This document describes the Machine Learning API endpoints for the WLASL-100 ASL recognition system.

## Base URL

```
http://localhost:8000/api/ml
```

## Authentication

All ML endpoints are **public** and do not require authentication. This allows guest users to test the translation functionality.

---

## Endpoints

### 1. Sign-to-Text Translation

Translate ASL signs from video to text.

**Endpoint:** `POST /api/ml/sign-to-text`

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `video` (file, required): Video file containing ASL signs
    - Supported formats: MP4, AVI, MOV, WEBM
    - Maximum size: 50MB (configurable via `MAX_VIDEO_SIZE_MB`)
    - Maximum duration: 30 seconds (configurable via `MAX_VIDEO_DURATION_SEC`)
  - `use_tta` (boolean, optional): Enable Test-Time Augmentation
    - Default: `true`
    - TTA improves accuracy by ~2-5% but increases processing time by 4x

**Response:** `200 OK`
```json
{
  "recognized_text": "hello",
  "glosses": ["hello", "help", "home", "have", "hearing"],
  "confidence": 0.856,
  "top5_predictions": [
    [0, 0.856],
    [12, 0.042],
    [45, 0.031],
    [67, 0.019],
    [23, 0.015]
  ],
  "frame_count": 87,
  "processing_time_ms": 342,
  "sign_language": "ASL"
}
```

**Response Fields:**
- `recognized_text` (string): Top prediction (most likely ASL gloss)
- `glosses` (string[]): Top 5 predictions as human-readable glosses
- `confidence` (float): Confidence score for top prediction (0.0 to 1.0)
- `top5_predictions` (array): Top 5 predictions as [class_index, confidence] pairs
- `frame_count` (int): Number of frames processed from video
- `processing_time_ms` (int): Total processing time in milliseconds
- `sign_language` (string): Always "ASL" for WLASL-100 model

**Error Responses:**

`400 Bad Request` - Invalid input
```json
{
  "detail": "Unsupported video format: .avi. Allowed formats: .mp4, .avi, .mov, .webm"
}
```

```json
{
  "detail": "File too large. Maximum size: 50MB"
}
```

```json
{
  "detail": "Unable to process video file"
}
```

`500 Internal Server Error` - Processing error
```json
{
  "detail": "Internal error during video processing"
}
```

**Example Request (curl):**
```bash
curl -X POST "http://localhost:8000/api/ml/sign-to-text" \
  -F "video=@recording.mp4" \
  -F "use_tta=true"
```

**Example Request (Python):**
```python
import requests

url = "http://localhost:8000/api/ml/sign-to-text"
files = {"video": open("recording.mp4", "rb")}
data = {"use_tta": "true"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Recognized: {result['recognized_text']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Example Request (JavaScript):**
```javascript
const formData = new FormData();
formData.append('video', videoBlob, 'recording.webm');
formData.append('use_tta', 'true');

const response = await fetch('http://localhost:8000/api/ml/sign-to-text', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Recognized:', result.recognized_text);
console.log('Confidence:', result.confidence);
```

---

### 2. Environment Validation

Validate recording environment before capturing video.

**Endpoint:** `POST /api/ml/validate-environment`

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `frame` (file, required): Webcam frame image
    - Supported formats: JPEG, PNG
    - Recommended size: 640x480 or higher

**Response:** `200 OK`
```json
{
  "background": {
    "status": "clear",
    "score": 0.08,
    "message": "Background is clear - excellent!"
  },
  "body": {
    "status": "visible",
    "landmarks_detected": {
      "left_hand": true,
      "right_hand": true,
      "arms": true,
      "torso": true
    },
    "message": "All body parts visible - perfect!"
  },
  "distance": {
    "status": "optimal",
    "shoulder_width_px": 280,
    "message": "Distance is optimal - perfect!"
  },
  "overall": "ready"
}
```

**Response Fields:**

**Background Validation:**
- `status` (string): "clear", "acceptable", or "cluttered"
  - `clear`: Edge density < 0.1 (ideal)
  - `acceptable`: Edge density 0.1-0.2 (usable)
  - `cluttered`: Edge density > 0.2 (not recommended)
- `score` (float): Edge density score (0.0 to 1.0)
- `message` (string): Human-readable feedback

**Body Validation:**
- `status` (string): "visible", "partial", or "not_visible"
  - `visible`: All required body parts detected
  - `partial`: Some body parts missing
  - `not_visible`: Insufficient body parts detected
- `landmarks_detected` (object): Boolean flags for each body part
  - `left_hand`: Left hand visible
  - `right_hand`: Right hand visible
  - `arms`: Both arms visible (shoulders to wrists)
  - `torso`: Torso visible (shoulders to hips)
- `message` (string): Human-readable feedback

**Distance Validation:**
- `status` (string): "optimal", "acceptable", or "out_of_range"
  - `optimal`: Shoulder width 200-350px (ideal)
  - `acceptable`: Shoulder width 150-200px or 350-400px (usable)
  - `out_of_range`: Shoulder width < 150px or > 400px (not recommended)
- `shoulder_width_px` (int): Measured shoulder width in pixels
- `message` (string): Human-readable feedback

**Overall Status:**
- `overall` (string): "ready" or "not_ready"
  - `ready`: All validation criteria met (background clear/acceptable, body visible, distance optimal/acceptable)
  - `not_ready`: One or more validation criteria failed

**Error Responses:**

`400 Bad Request` - Invalid input
```json
{
  "detail": "Unsupported image format: .bmp. Allowed formats: .jpg, .jpeg, .png"
}
```

```json
{
  "detail": "Unable to decode image"
}
```

`500 Internal Server Error` - Processing error
```json
{
  "detail": "Internal error during environment validation"
}
```

**Example Request (curl):**
```bash
curl -X POST "http://localhost:8000/api/ml/validate-environment" \
  -F "frame=@webcam_frame.jpg"
```

**Example Request (JavaScript):**
```javascript
// Capture frame from video element
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0);

// Convert to blob
const blob = await new Promise(resolve => {
  canvas.toBlob(resolve, 'image/jpeg', 0.8);
});

// Send to validation endpoint
const formData = new FormData();
formData.append('frame', blob, 'frame.jpg');

const response = await fetch('http://localhost:8000/api/ml/validate-environment', {
  method: 'POST',
  body: formData
});

const validation = await response.json();
console.log('Overall status:', validation.overall);
console.log('Background:', validation.background.status);
console.log('Body:', validation.body.status);
console.log('Distance:', validation.distance.status);
```

---

### 3. Service Statistics

Get ML service statistics and performance metrics.

**Endpoint:** `GET /api/ml/stats`

**Request:** No parameters required

**Response:** `200 OK`
```json
{
  "total_predictions": 42,
  "average_inference_time_ms": 287.5,
  "average_confidence": 0.823,
  "model_loaded": true,
  "device": "cuda"
}
```

**Response Fields:**
- `total_predictions` (int): Total number of predictions made since server start
- `average_inference_time_ms` (float): Average inference time in milliseconds
- `average_confidence` (float): Average confidence score across all predictions
- `model_loaded` (bool): Whether the model is loaded and ready
- `device` (string): Device used for inference ("cuda" or "cpu")

**Example Request (curl):**
```bash
curl -X GET "http://localhost:8000/api/ml/stats"
```

---

### 4. Health Check

Check if ML service is healthy and ready.

**Endpoint:** `GET /api/ml/health`

**Request:** No parameters required

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

**Response Fields:**
- `status` (string): "healthy" or "unhealthy"
- `model_loaded` (bool): Whether the model is loaded
- `device` (string): Device used for inference

**Example Request (curl):**
```bash
curl -X GET "http://localhost:8000/api/ml/health"
```

---

## Performance Benchmarks

### GPU (NVIDIA RTX 3070, CUDA 11.8)

| Configuration | Inference Time | Throughput |
|---------------|----------------|------------|
| TTA Enabled   | 200-500ms      | 2-5 videos/sec |
| TTA Disabled  | 50-125ms       | 8-20 videos/sec |

### CPU (Intel i7-10700K)

| Configuration | Inference Time | Throughput |
|---------------|----------------|------------|
| TTA Enabled   | 1-2s           | 0.5-1 videos/sec |
| TTA Disabled  | 250-500ms      | 2-4 videos/sec |

**Notes:**
- TTA (Test-Time Augmentation) improves accuracy by ~2-5% but increases processing time by 4x
- GPU is recommended for production use (5-10x faster than CPU)
- Processing time includes keypoint extraction, sequence preprocessing, and inference
- Actual performance may vary based on video length, resolution, and hardware

---

## Error Handling

All endpoints follow consistent error response format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Request successful
- `400 Bad Request`: Invalid input (wrong format, file too large, etc.)
- `500 Internal Server Error`: Server-side processing error

**Graceful Degradation:**
- GPU out-of-memory errors automatically fall back to CPU
- MediaPipe extraction failures use zero-filled features
- Partial validation results returned when possible

---

## Rate Limiting

Currently, there is no rate limiting on ML endpoints. For production deployment, consider:
- Rate limiting per IP address (e.g., 10 requests/minute)
- Request queuing for high load scenarios
- Caching validation results for repeated frames

---

## CORS Configuration

CORS is enabled for frontend access. Default configuration:
```python
allow_origins=["http://localhost:3000"]  # Frontend origin
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

For production, update `allow_origins` to your frontend domain.

---

## Testing

### Manual Testing with curl

**Test sign-to-text:**
```bash
# Record a video with ASL signs
# Upload for translation
curl -X POST "http://localhost:8000/api/ml/sign-to-text" \
  -F "video=@test_video.mp4" \
  -F "use_tta=true"
```

**Test environment validation:**
```bash
# Capture a webcam frame
# Validate environment
curl -X POST "http://localhost:8000/api/ml/validate-environment" \
  -F "frame=@test_frame.jpg"
```

**Test service stats:**
```bash
curl -X GET "http://localhost:8000/api/ml/stats"
```

### Automated Testing

See `emotisign-backend/tests/` for unit and integration tests.

```bash
# Run all tests
pytest

# Run ML API tests only
pytest tests/test_ml_api.py

# Run with coverage
pytest --cov=app/ml --cov=app/routers
```

---

## Troubleshooting

### "Model not loaded" error
- Verify model files exist in `app/ml/models/wlasl100/`
- Check `MODEL_PATH` in `.env`
- Run `python health_check.py` to diagnose

### "CUDA out of memory" error
- Reduce video resolution or length
- Disable TTA (`use_tta=false`)
- System will automatically fall back to CPU

### "MediaPipe initialization failed" error
- Ensure mediapipe==0.9.3 (newer versions have breaking changes)
- On Windows, install Visual C++ Redistributable

### Slow inference on CPU
- Enable GPU acceleration (see README for CUDA setup)
- Disable TTA for faster inference
- Reduce video length

### "Unable to process video file" error
- Verify video format is supported (MP4, AVI, MOV, WEBM)
- Check video is not corrupted
- Ensure video contains actual frames

---

## Support

For issues or questions:
- Check the [README](../README.md) for setup instructions
- Run `python health_check.py` to diagnose issues
- Review logs in console output
- Contact the development team
