# EmotiSign — FastAPI Backend

> **Final Year Project | University of Central Punjab | Group F25SE007**
> Bidirectional Sign Language Communication with Emotion Detection

---

## 🏗️ Project Structure

```
emotisign-backend/
├── main.py                         # FastAPI app entry point
├── requirements.txt
├── .env                            # Environment variables
├── app/
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── database.py                 # SQLAlchemy async engine + session
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── schemas.py                  # Pydantic request/response schemas
│   ├── dependencies.py             # FastAPI dependencies (auth injection)
│   ├── routers/
│   │   ├── auth.py                 # /api/auth — register, login, me
│   │   ├── translation.py          # /api/translate — text-to-sign, sign-to-text
│   │   └── chat.py                 # /api/chat — rooms, messages, WebSocket
│   ├── services/
│   │   ├── auth_service.py         # JWT + password hashing
│   │   └── ws_manager.py           # WebSocket connection manager
│   └── ml/
│       └── ml_service.py           # 🤖 ML placeholder (ready for real models)
├── uploads/
│   └── videos/                     # Uploaded sign language videos
└── static/
    └── signs/                      # Sign GIFs / avatar frames
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Edit `.env`:
```env
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=60
DATABASE_URL=sqlite+aiosqlite:///./emotisign.db
UPLOAD_DIR=uploads/videos
MAX_VIDEO_SIZE_MB=50
```

### 3. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API docs

- Swagger UI: http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc

---

## 🔑 Authentication API — `/api/auth`

| Method | Endpoint         | Auth? | Description                  |
|--------|------------------|-------|------------------------------|
| POST   | `/register`      | ❌    | Register new user            |
| POST   | `/login`         | ❌    | Login → JWT token            |
| GET    | `/me`            | ✅    | Get current user profile     |
| PATCH  | `/me`            | ✅    | Update profile               |
| POST   | `/logout`        | ✅    | Logout (client discards JWT) |

### Register Example
```json
POST /api/auth/register
{
  "username": "hamza",
  "email": "hamza@example.com",
  "password": "securepass123",
  "full_name": "Muhammad Hamza",
  "role": "hearing",
  "preferred_sign_language": "ASL"
}
```

### Login Example
```json
POST /api/auth/login
{
  "username": "hamza",
  "password": "securepass123"
}
// Returns: { "access_token": "...", "token_type": "bearer", "expires_in": 3600 }
```

**Use the token** as: `Authorization: Bearer <token>`

---

## 🤟 Translation API — `/api/translate`

### Text → Sign
```
POST /api/translate/text-to-sign
Authorization: Bearer <token>   (optional — guests allowed)

{
  "text": "Hello, how are you?",
  "sign_language": "ASL",
  "include_emotion": true
}
```

**Response includes:**
- `signs[]` — list of sign units with frames and GIF URLs
- `fingerspelled_words[]` — OOV words that were fingerspelled
- `emotion_analysis` — sentiment + emotion detected from the text
- `processing_time_ms`

---

### Sign → Text (video upload)
```
POST /api/translate/sign-to-text
Authorization: Bearer <token>   (optional — guests allowed)
Content-Type: multipart/form-data

Fields:
  video: <file>        (mp4, webm, avi, mov — max 50MB)
  sign_language: ASL
  detect_emotion: true
```

**Response includes:**
- `recognized_text` — transcribed sentence
- `glosses[]` — raw ASL sign glosses
- `confidence` — model confidence score
- `emotion_from_video` — facial expression emotion from video

---

### Translation History
```
GET    /api/translate/history?page=1&page_size=20&mode=text_to_sign
GET    /api/translate/history/{id}
DELETE /api/translate/history/{id}
```

---

## 💬 Chat API — `/api/chat`

### REST Endpoints

| Method | Endpoint                         | Description              |
|--------|----------------------------------|--------------------------|
| POST   | `/rooms`                         | Create a chat room       |
| GET    | `/rooms`                         | List my rooms            |
| GET    | `/rooms/{id}/messages`           | Get message history      |
| PATCH  | `/rooms/{id}/read`               | Mark messages as read    |

### Create Room Example
```json
POST /api/chat/rooms
{
  "name": "Hamza-Asad DM",
  "member_ids": [2, 3],
  "is_direct": true
}
```

---

## 🔌 WebSocket Chat

Connect to:
```
ws://localhost:8000/api/chat/ws/chat/{room_id}?token=<JWT>
```

### Client → Server Events (send JSON)

```json
// Send a text message
{ "type": "text", "content": "Hello!", "auto_translate": true }

// Send a sign video URL (for Sign→Text auto-translation)
{ "type": "sign_video_url", "video_url": "/uploads/vid.mp4", "auto_translate": true }

// Typing indicator
{ "type": "typing" }

// Mark messages as read
{ "type": "read" }

// Keepalive ping
{ "type": "ping" }
```

### Server → Client Events

```json
// New message
{ "event": "message", "data": { "id": 1, "text_content": "Hello!", "emotion": "happy", ... }, "timestamp": "..." }

// Someone is typing
{ "event": "typing", "data": { "user_id": 2, "username": "asad" }, "timestamp": "..." }

// User joined/left
{ "event": "user_joined", "data": { "user_id": 2, "online_users": [1, 2] }, "timestamp": "..." }
{ "event": "user_left",   "data": { "user_id": 2, "online_users": [1] },    "timestamp": "..." }

// Online users (sent on connect)
{ "event": "online_users", "data": { "users": [1, 2] }, "timestamp": "..." }

// Read receipt
{ "event": "read", "data": { "user_id": 2, "room_id": 1 }, "timestamp": "..." }

// Pong response
{ "event": "pong", "data": {}, "timestamp": "..." }

// Error
{ "event": "error", "data": { "detail": "Invalid JSON" }, "timestamp": "..." }
```

---

## 🗄️ Database Models

| Table                | Purpose                                        |
|----------------------|------------------------------------------------|
| `users`              | User accounts (deaf / hearing / admin)         |
| `translation_history`| All text-to-sign and sign-to-text translations |
| `chat_rooms`         | Chat rooms (DMs and groups)                    |
| `chat_room_members`  | Room membership                                |
| `chat_messages`      | All messages (text + sign video)               |
| `user_sessions`      | JWT session tracking (for future revocation)   |

---

## 🤖 ML Integration Points

All ML functions are in `app/ml/ml_service.py`. Each function has a docstring explaining exactly what the real implementation should do.

| Function                    | Placeholder? | Real Integration                            |
|-----------------------------|-------------|---------------------------------------------|
| `text_to_sign()`            | ✅ Yes      | Sign dictionary + avatar/GIF renderer       |
| `sign_to_text()`            | ✅ Yes      | MediaPipe + LSTM/Transformer on video       |
| `analyze_sentiment()`       | ✅ Yes      | VADER / RoBERTa sentiment model             |
| `detect_emotion_from_video()` | ✅ Yes    | DeepFace / FER facial expression model      |

To plug in a real model: replace the function body while keeping the return shape identical.

---

## 🧠 WLASL-100 Model Setup

### Dependencies Installation

The WLASL-100 ASL recognition model requires PyTorch, MediaPipe, and OpenCV.

#### Option 1: CPU-only (for development/testing)
```bash
pip install -r requirements.txt
```

#### Option 2: GPU-accelerated (recommended for production)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- CUDA 11.8 installed ([Download CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive))
- cuDNN 8.x for CUDA 11.8

**Install PyTorch with CUDA 11.8:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**Then install other dependencies:**
```bash
pip install mediapipe==0.9.3 opencv-python==4.8.1.78 numpy==1.24.3
```

### Model Files Setup

The model files should be located in `app/ml/models/wlasl100/`:

```
app/ml/models/wlasl100/
├── best_model.pth        # Trained model checkpoint (25MB)
├── vocab.json            # 100-word vocabulary mapping
└── temperature.json      # Temperature scaling for confidence calibration
```

These files are already included in the repository.

### Verify Installation

Run the health check script to verify everything is set up correctly:

```bash
python health_check.py
```

Expected output:
```
=== EmotiSign ML Service Health Check ===

✓ Found: app/ml/models/wlasl100/best_model.pth
✓ Found: app/ml/models/wlasl100/vocab.json
✓ GPU available: NVIDIA GeForce RTX 3070
  CUDA version: 11.8
  GPU memory: 8.00 GB
✓ MediaPipe initialized successfully

✓ All checks passed
```

### Environment Variables

Configure ML settings in `.env`:

```env
# ML Model Configuration
MODEL_PATH=app/ml/models/wlasl100/best_model.pth
VOCAB_PATH=app/ml/models/wlasl100/vocab.json
TEMPERATURE_PATH=app/ml/models/wlasl100/temperature.json
ML_DEVICE=cuda  # or 'cpu' for CPU-only inference

# Inference Configuration
USE_TTA=true
CONFIDENCE_THRESHOLD=0.25
NUM_FRAMES=64

# Video Processing
MAX_VIDEO_SIZE_MB=50
MAX_VIDEO_DURATION_SEC=30
UPLOAD_DIR=uploads/videos
```

### Troubleshooting

**GPU not detected:**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with correct CUDA version

**MediaPipe errors:**
- Ensure mediapipe==0.9.3 (newer versions have breaking changes)
- On Windows, may need Visual C++ Redistributable

**Out of memory errors:**
- Reduce batch size or disable TTA
- System will automatically fall back to CPU if GPU OOM occurs

---

## 📦 Tech Stack

| Layer      | Technology                                    |
|------------|-----------------------------------------------|
| Framework  | FastAPI 0.111                                 |
| Database   | SQLite via SQLAlchemy Async + aiosqlite       |
| Auth       | JWT (python-jose) + bcrypt (passlib)          |
| WebSocket  | FastAPI WebSocket (Starlette)                 |
| Validation | Pydantic v2                                   |
| File I/O   | aiofiles (async)                              |
| ML (mock)  | Async placeholder functions in ml_service.py  |

---

## 🔐 Security Notes (Production Checklist)

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Restrict `allow_origins` in CORS to your frontend domain
- [ ] Use PostgreSQL instead of SQLite for concurrent connections
- [ ] Add HTTPS / TLS termination (Nginx/Traefik)
- [ ] Implement JWT blocklist for logout (Redis recommended)
- [ ] Add rate limiting (slowapi)
- [ ] Validate file content (not just MIME type) for video uploads
