# Sign Language Video Validation Guide

EmotiSign automatically validates all uploaded videos to ensure they contain actual sign language content. This prevents processing of irrelevant videos and improves recognition accuracy.

## What Videos Are Accepted?

Videos must meet **all** of these criteria:

### ✅ Content Requirements

1. **Hand Visibility**
   - At least **30% of frames** must show clearly visible hands
   - Both hands should be in frame when signing
   - Hands should be clearly separated from the background

2. **Duration**
   - Minimum: **0.5 seconds**
   - Maximum: **30 seconds**
   - Single-sign videos should be 2-5 seconds
   - Multi-word sentences can be longer

3. **Motion**
   - Video must show actual movement (not a static image)
   - Sign language involves dynamic hand/arm motion

### ✅ Technical Requirements

1. **Resolution**
   - Minimum: **320×240 pixels** (240p)
   - Recommended: **640×480 or higher** (480p+)

2. **File Format**
   - MP4, WebM, AVI, MOV, or QuickTime

3. **File Size**
   - Maximum: **50 MB**

### ✅ Recording Best Practices

#### Camera Position
- **Chest/torso level** — not face-level or waist-level
- Shows from waist to above the head
- Signer centered in frame

#### Lighting
- **Well-lit environment** — avoid dim rooms
- Even lighting (no harsh shadows or backlight)
- Front-facing light source is best

#### Background
- **Plain, uncluttered background**
- Contrasting with skin tone (e.g., light wall for darker skin)
- Avoid busy patterns or moving objects

#### Framing
- **Full upper body visible**
- Arms should not go out of frame during signing
- Comfortable distance (not too close or too far)

---

## Common Validation Errors

### ❌ "Insufficient hand visibility detected"

**Problem:** Less than 30% of frames show visible hands.

**Causes:**
- Hands moving out of frame frequently
- Poor lighting — hands not distinguishable
- Complex background confusing hand detection
- Hands occluded by objects or body

**Fix:**
- Ensure hands stay in frame during entire sign
- Improve lighting (add a desk lamp or face a window)
- Use a plain background (hang a bedsheet if needed)
- Move camera back to capture full signing space

### ❌ "Video too short / too long"

**Problem:** Duration outside 0.5–30 second range.

**Fix:**
- For single signs: 2-5 seconds is ideal
- For sentences: keep under 30 seconds
- Trim videos to remove long pauses before/after signing

### ❌ "No motion detected"

**Problem:** Video is a static image or has no movement.

**Fix:**
- Actually perform the sign (don't hold a static pose)
- Check that video file is not corrupted
- Ensure recording is not frozen

### ❌ "Video resolution too low"

**Problem:** Video is under 320×240 pixels.

**Fix:**
- Use webcam recording (built-in cameras are usually 480p+)
- Check camera settings for higher resolution
- Don't upload heavily compressed/downscaled videos

---

## Supported Sign Languages

### ASL (American Sign Language)
- **Model:** WLASL-100 — recognizes 100 common ASL words
- **Mode:** Upload a video of a single ASL sign
- **Auto-translation:** Yes — AI translates sign → English text

### PSL (Pakistan Sign Language)
- **Model:** 37-letter alphabet classifier
- **Mode:** Upload a video + add a text label
- **Manual labeling:** Required — enter the Urdu letter or meaning

---

## For FYP Evaluators

### Testing ASL Recognition
1. Record a video of any of the [100 WLASL signs](../app/ml/models/wlasl100/vocab.json)
2. Upload via **Translate → Sign-to-Text** or **Chat**
3. System validates hands are visible, then runs AI recognition
4. Result shows recognized word + confidence score

### Testing PSL Alphabet
1. Record a PSL alphabet letter (ا، ب، ت، etc.)
2. Use **Chat → PSL video** or **Translate → PSL Alphabet (live)**
3. System validates video, then you add the label
4. Label is sent along with video — no AI translation for PSL

### Intentional Failures (Edge Cases)
Try these to see validation in action:
- Upload a video with no hands visible → **rejected**
- Upload a static image renamed as .mp4 → **rejected**
- Upload a 60-second video → **rejected (too long)**
- Upload a video with hands only visible for 10% of frames → **rejected**

All rejection messages are user-friendly and explain what went wrong.

---

## Developer Notes

### Implementation

Validation runs in `app/services/video_validator.py`:

```python
from app.services.video_validator import validate_sign_video_bytes, VideoValidationError

try:
    validation_result = validate_sign_video_bytes(content, filename)
    print(f"✅ Video validation passed: {validation_result}")
except VideoValidationError as e:
    raise HTTPException(status_code=400, detail=str(e))
```

### Validation Process

1. **File I/O Check** — video can be opened and has frames
2. **Duration Check** — 0.5s ≤ duration ≤ 30s
3. **Resolution Check** — width ≥ 320, height ≥ 240
4. **Hand Detection** — MediaPipe Hands detects hands in ≥30% of frames
5. **Motion Check** — frame-to-frame difference shows movement

Validation samples **30 frames uniformly** across the video (not every frame) for performance.

### Customization

Adjust thresholds in `video_validator.py` constructor:

```python
SignLanguageVideoValidator(
    min_duration_sec=0.5,        # Minimum video length
    max_duration_sec=30.0,       # Maximum video length
    min_hand_frames_ratio=0.30,  # 30% of frames need hands
    sample_frame_count=30,       # Frames to check
)
```

For stricter validation (e.g., require 50% hand visibility):
```python
min_hand_frames_ratio=0.50
```

For lenient testing (accept videos with 15% hand visibility):
```python
min_hand_frames_ratio=0.15
```

---

## Troubleshooting

### "My valid video is being rejected!"

1. **Check lighting** — record a test video in a well-lit room
2. **Check framing** — ensure hands are visible throughout
3. **Check background** — use a plain wall as background
4. **Check motion** — ensure you're actually signing (moving hands)

If validation still fails, check the **backend logs** for detailed metrics:
```
✅ Video validation passed: {'duration_sec': 3.2, 'hand_visibility_ratio': 0.867, ...}
```

### "Validation is too strict"

You can lower the threshold by setting an environment variable:

```bash
# In .env file
VIDEO_VALIDATION_MIN_HAND_RATIO=0.20  # Accept 20% instead of 30%
```

Then update the validator initialization in `video_validator.py`.

---

## Future Improvements

- **Face detection** — require face visible for emotion analysis
- **Body pose check** — ensure full upper body is in frame
- **Blur detection** — reject out-of-focus videos
- **Lighting analysis** — reject under/overexposed videos
- **FPS check** — warn if video is <15fps (choppy motion)
