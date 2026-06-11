# Video Validation — Quick Reference Card

## ✅ What Videos Pass

- **Hands visible** in ≥30% of frames
- **Duration** between 0.5–30 seconds
- **Resolution** ≥320×240 pixels
- **Motion detected** (not static)
- **Format:** MP4, WebM, AVI, MOV
- **Size** ≤50 MB

## ❌ Common Rejections

| Error | Fix |
|---|---|
| "Insufficient hand visibility" | Better lighting + plain background |
| "No motion detected" | Record actual signing (not static pose) |
| "Video too short/long" | 2-5 seconds for single signs |
| "Resolution too low" | Use webcam (don't downscale) |

## 🎥 Recording Tips

1. **Position:** Chest level, shows waist→head
2. **Lighting:** Front-lit, no shadows
3. **Background:** Plain wall
4. **Framing:** Hands stay in frame

## 🧪 Testing (FYP Evaluators)

**Valid:** Sign "hello" for 3s → ✅ Passes  
**Invalid:** Video of desk → ❌ "No hands visible"  
**Invalid:** Static photo → ❌ "No motion"  
**Invalid:** 60s video → ❌ "Too long"  

## 🔧 Quick Disable (Dev Only)

Comment out in `translation.py`, `ml_router.py`, `chat.py`:
```python
# try:
#     validate_sign_video_bytes(content, filename)
# except VideoValidationError as e:
#     raise HTTPException(...)
```

## 📊 Check Validation Metrics

Look for backend log:
```
✅ Video validation passed: {
  'hand_visibility_ratio': 0.867,  ← Must be ≥0.30
  'duration_sec': 3.2,
  'motion_detected': True
}
```
