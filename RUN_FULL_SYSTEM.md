# 🚀 Running Backend + Frontend Together

Complete guide to run the full EmotiSign system with WLASL-100 model integration.

---

## Prerequisites

- ✅ Python 3.9/3.10 installed
- ✅ Node.js 16+ installed
- ✅ npm or yarn installed

---

## Step-by-Step Guide

### Step 1: Setup Backend

#### 1.1 Install Backend Dependencies

**Windows:**
```cmd
cd emotisign-backend
install.bat
```

**Linux/Mac:**
```bash
cd emotisign-backend
chmod +x install.sh
./install.sh
```

#### 1.2 Verify Backend Installation

```bash
python health_check.py
```

Should show: `✓ All checks passed!`

#### 1.3 Test Backend

```bash
python run_tests.py
```

Should show: `✓ All tests passed!`

---

### Step 2: Setup Frontend

#### 2.1 Install Frontend Dependencies

```bash
cd ../emotisign-frontend
npm install
```

Or with yarn:
```bash
yarn install
```

#### 2.2 Configure Frontend Environment

Create or verify `.env.local` file:

```bash
# emotisign-frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### Step 3: Start Backend Server

Open **Terminal 1** (keep it running):

```bash
cd emotisign-backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ **Backend is running!**

---

### Step 4: Start Frontend Server

Open **Terminal 2** (keep it running):

```bash
cd emotisign-frontend
npm run dev
```

Or with yarn:
```bash
yarn dev
```

Expected output:
```
  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
  - Network:      http://192.168.x.x:3000

 ✓ Ready in 2.5s
```

✅ **Frontend is running!**

---

### Step 5: Test the System

#### 5.1 Test Backend API

In **Terminal 3**:

```bash
# Health check
curl http://localhost:8000/api/ml/health

# Stats
curl http://localhost:8000/api/ml/stats
```

#### 5.2 Test Frontend

Open browser: **http://localhost:3000**

Navigate to: **http://localhost:3000/translate/sign-to-text**

---

### Step 6: Test Complete Workflow

1. **Open**: http://localhost:3000/translate/sign-to-text

2. **Click "Start Validation"**
   - Grant webcam permissions when prompted

3. **Wait for validation** (real-time checks):
   - ✅ Background: Clear/Acceptable
   - ✅ Body Visibility: Visible
   - ✅ Distance: Optimal/Acceptable

4. **Click "Start Recording"** (when button is enabled)
   - Button only enables when all validation checks pass

5. **Perform an ASL sign** (2-5 seconds)
   - Keep hands visible
   - Face the camera
   - Use clear background

6. **Click "Stop Recording"**

7. **Wait for processing** (~1-2 seconds on CPU, ~200-500ms on GPU)

8. **View results**:
   - Recognized text (top prediction)
   - Confidence score
   - Top 5 predictions with confidence bars
   - Processing metadata

9. **Click "Record Again"** to test another sign

---

## 🖥️ Terminal Setup

You need **2 terminals running simultaneously**:

### Terminal 1: Backend
```bash
cd emotisign-backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2: Frontend
```bash
cd emotisign-frontend
npm run dev
```

**Keep both terminals running!**

---

## 🌐 URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main application |
| **Sign-to-Text** | http://localhost:3000/translate/sign-to-text | ASL translation page |
| **Backend API** | http://localhost:8000 | API endpoints |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Health Check** | http://localhost:8000/api/ml/health | Backend health status |

---

## 🧪 Testing the Integration

### Test 1: Backend Health

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

### Test 2: Frontend Loads

Open: http://localhost:3000/translate/sign-to-text

Should see:
- Video feed placeholder
- "Start Validation" button
- Sign language selector (ASL/PSL)

### Test 3: Validation Works

1. Click "Start Validation"
2. Grant webcam permissions
3. See real-time validation feedback:
   - Background status
   - Body visibility status
   - Distance status

### Test 4: Recording Works

1. Wait for "Start Recording" button to enable
2. Click "Start Recording"
3. See recording indicator (red dot + timer)
4. Click "Stop Recording"
5. See processing overlay

### Test 5: Results Display

After processing:
- See recognized text
- See confidence score
- See top 5 predictions
- See processing time

---

## 🐛 Troubleshooting

### Issue: "Backend not responding"

**Check:**
```bash
curl http://localhost:8000/api/ml/health
```

**Solution:**
- Verify backend is running in Terminal 1
- Check for errors in backend terminal
- Restart backend: `uvicorn main:app --reload`

### Issue: "Frontend can't connect to backend"

**Check `.env.local`:**
```bash
# emotisign-frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Solution:**
- Verify URL is correct
- Restart frontend: `npm run dev`
- Check CORS is enabled in backend

### Issue: "Webcam not accessible"

**Solution:**
- Grant camera permissions in browser
- Close other apps using webcam
- Try different browser (Chrome recommended)
- Check browser console for errors

### Issue: "Validation stuck at 'not_ready'"

**Possible causes:**
- Background too cluttered → Use plain background
- Body not visible → Ensure full upper body in frame
- Distance wrong → Move closer/farther from camera

**Check validation feedback messages for specific guidance**

### Issue: "Upload fails"

**Check:**
- Video file size < 50MB
- Recording duration < 30 seconds
- Backend is running
- Check browser console for errors

### Issue: "Port already in use"

**Backend (port 8000):**
```bash
uvicorn main:app --reload --port 8001
```

Then update frontend `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8001
```

**Frontend (port 3000):**
```bash
npm run dev -- -p 3001
```

---

## ⚙️ Configuration

### Backend Configuration (`.env`)

```env
# ML Model
MODEL_PATH=app/ml/models/wlasl100/best_model.pth
VOCAB_PATH=app/ml/models/wlasl100/vocab.json
ML_DEVICE=cpu  # or 'cuda' for GPU

# Inference
USE_TTA=true
CONFIDENCE_THRESHOLD=0.25

# Video Limits
MAX_VIDEO_SIZE_MB=50
MAX_VIDEO_DURATION_SEC=30
```

### Frontend Configuration (`.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📊 Performance

### CPU (Intel i7)
- Validation: Real-time (5 fps)
- Recording: 30 fps
- Processing: 1-2 seconds per video
- Total workflow: ~5-10 seconds

### GPU (NVIDIA RTX 3070)
- Validation: Real-time (5 fps)
- Recording: 30 fps
- Processing: 200-500ms per video
- Total workflow: ~3-5 seconds

---

## 🎯 Complete Workflow Example

```
1. User opens http://localhost:3000/translate/sign-to-text
   ↓
2. User clicks "Start Validation"
   ↓
3. Frontend captures frames (5 fps)
   ↓
4. Frontend sends frames to backend /api/ml/validate-environment
   ↓
5. Backend validates: background, body, distance
   ↓
6. Frontend displays validation feedback
   ↓
7. When ready, user clicks "Start Recording"
   ↓
8. Frontend records video using MediaRecorder
   ↓
9. User clicks "Stop Recording"
   ↓
10. Frontend uploads video to backend /api/ml/sign-to-text
    ↓
11. Backend processes video:
    - Extract keypoints with MediaPipe
    - Resample to 64 frames
    - Generate TTA variants
    - Run inference
    - Apply temperature scaling
    ↓
12. Backend returns results
    ↓
13. Frontend displays:
    - Recognized text
    - Confidence score
    - Top 5 predictions
    - Processing time
```

---

## 🔄 Restart Services

### Restart Backend:
```bash
# In Terminal 1, press CTRL+C
# Then restart:
uvicorn main:app --reload
```

### Restart Frontend:
```bash
# In Terminal 2, press CTRL+C
# Then restart:
npm run dev
```

---

## 📝 Quick Commands Reference

### Start Both Services:

**Terminal 1 (Backend):**
```bash
cd emotisign-backend
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd emotisign-frontend
npm run dev
```

### Test Backend:
```bash
curl http://localhost:8000/api/ml/health
```

### Test Frontend:
Open: http://localhost:3000/translate/sign-to-text

### Stop Services:
Press **CTRL+C** in each terminal

---

## ✅ Success Checklist

- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Backend health check passed
- [ ] Backend tests passed
- [ ] Backend server running (Terminal 1)
- [ ] Frontend server running (Terminal 2)
- [ ] Backend API responding
- [ ] Frontend loads in browser
- [ ] Webcam access granted
- [ ] Validation works
- [ ] Recording works
- [ ] Upload works
- [ ] Results display correctly

---

## 🎉 You're Ready!

Both backend and frontend are now running. Test the complete workflow:

1. Open: http://localhost:3000/translate/sign-to-text
2. Start validation
3. Record ASL sign
4. View translation results

**Enjoy using EmotiSign! 🤟**

---

## 📚 Additional Resources

- **Backend Setup**: `emotisign-backend/SETUP_GUIDE.md`
- **API Documentation**: `emotisign-backend/docs/api.md`
- **Troubleshooting**: `emotisign-backend/docs/troubleshooting.md`
- **Frontend README**: `emotisign-frontend/README.md`
