# Video Recording and Display Fixes — Bugfix Design

## Overview

This document covers the design for three distinct bugs in the sign language translation application:

1. **WLASL Frame Trimming**: Starting and ending transition frames (user getting into position / relaxing after the sign) are included in the video sent for WLASL inference, degrading prediction accuracy. The fix trims the first and last N frames in `wlasl_service.py` before keypoint extraction, but only for webcam-recorded videos (not user-uploaded files).

2. **Auto-Stop Recording**: Users must manually click "Stop Recording". The fix adds a presence-detection loop in the frontend (`page.tsx`) that monitors body-landmark absence during recording and calls the existing `stopRecording()` function after a sustained 1–2 second absence.

3. **PSL Urdu Text Garbled**: The `label_map.json` file contains mojibake (UTF-8 Urdu characters misread as Latin-1), and the frontend has no Urdu-capable font. The fix re-encodes the label map correctly and adds a Google Fonts import for Noto Nastaliq Urdu.

---

## Glossary

- **Bug_Condition (C)**: The specific input condition that triggers each bug.
- **Property (P)**: The desired correct behavior when the bug condition holds.
- **Preservation**: Existing behaviors that must remain unchanged after the fix.
- **isBugCondition(input)**: Pseudocode predicate that returns `true` when the bug is triggered.
- **`extract_keypoints_from_video`**: Method in `wlasl_service.py` that reads a video file frame-by-frame and returns a `(N, 126)` keypoint array.
- **`sign_to_text`**: Top-level method in `WLASLModelService` that orchestrates extraction → inference.
- **`stopRecording`**: Frontend callback in `page.tsx` that captures a thumbnail and calls `mediaRecorder.stop()`, transitioning to the `'review'` state.
- **`label_map.json`**: JSON file at `app/ml/models/psl/label_map.json` mapping integer class indices to Urdu character strings.
- **Mojibake**: Garbled text produced when UTF-8 bytes are decoded as a different encoding (e.g., Latin-1/Windows-1252).
- **`PSLLiveService.predict`**: Async method that processes a single BGR frame and returns a prediction dict including `urdu_text`.
- **`TRIM_FRAMES`**: Configurable constant for the number of frames to remove from each end of a webcam-recorded video (proposed default: 10 frames ≈ 0.33 s at 30 fps).
- **`ABSENCE_THRESHOLD_MS`**: Frontend constant for how long (ms) the user must be absent before auto-stop triggers (proposed default: 1500 ms).

---

## Bug Details

### Bug 1 — WLASL Starting/Ending Frame Trimming

#### Bug Condition

The bug manifests when a user records a video through the app's webcam flow and submits it for WLASL prediction. The `sign_to_text` pipeline processes every frame including the initial frames where the user is moving into position and the final frames where the user is relaxing, which shifts the keypoint sequence away from the actual sign.

**Formal Specification:**
```
FUNCTION isBugCondition_1(input)
  INPUT: input = { video_path: str, source: "webcam" | "upload" }
  OUTPUT: boolean

  RETURN input.source == "webcam"
         AND totalFrames(input.video_path) > (2 * TRIM_FRAMES)
         AND firstNFrames(input.video_path, TRIM_FRAMES) contain transition motion
         AND lastNFrames(input.video_path, TRIM_FRAMES) contain transition motion
END FUNCTION
```

#### Examples

- User presses "Start", spends 0.5 s moving hands into position, signs "hello", then lowers hands → first ~15 and last ~15 frames are transition frames; model sees a blended sequence and predicts incorrectly.
- User records a 3-second sign at 30 fps (90 frames); trimming 10 frames from each end leaves 70 frames of clean sign content.
- User uploads a pre-recorded, already-trimmed video → trimming must NOT be applied (upload path is unaffected).
- Video is very short (≤ 2 × TRIM_FRAMES after recording) → trimming is skipped; padding in `resample_sequence` handles the short sequence.

---

### Bug 2 — Manual Recording Stop Required

#### Bug Condition

The bug manifests during the `'recording'` state when the user completes their sign and moves out of frame. The `MediaRecorder` continues running, accumulating empty/background-only frames until the user manually clicks "Stop Recording" or the 30-second timeout fires.

**Formal Specification:**
```
FUNCTION isBugCondition_2(input)
  INPUT: input = { state: PageState, bodyLandmarksDetected: boolean, absenceDurationMs: number }
  OUTPUT: boolean

  RETURN input.state == "recording"
         AND input.bodyLandmarksDetected == false
         AND input.absenceDurationMs >= ABSENCE_THRESHOLD_MS
END FUNCTION
```

#### Examples

- User signs "book", lowers hands and steps back → body landmarks absent for 1.5 s → recording auto-stops and transitions to `'review'`.
- User briefly turns their head during a sign (< 1.5 s absence) → recording continues uninterrupted.
- User manually clicks "Stop Recording" at any time → existing immediate-stop behavior is unchanged.
- Recording reaches 30-second maximum → existing timeout behavior is unchanged.

---

### Bug 3 — PSL Urdu Text Garbled

#### Bug Condition

The bug manifests when the PSL live service loads `label_map.json` and returns a prediction. The JSON file was saved with UTF-8 Urdu characters but was subsequently corrupted (mojibake): each Urdu code point was encoded as UTF-8 bytes and then those bytes were interpreted as Latin-1, producing sequences like `"+¡GÇ¼"` instead of the correct Urdu character.

**Formal Specification:**
```
FUNCTION isBugCondition_3(input)
  INPUT: input = { label_map_entry: str }
  OUTPUT: boolean

  RETURN containsMojibake(input.label_map_entry)
         // i.e., the string contains Latin-1 sequences that are
         // actually UTF-8 multi-byte sequences for Urdu Unicode code points
END FUNCTION
```

#### Examples

- `label_map.json` entry `"1": "+¡GÇ¼"` → should be a single Urdu character (e.g., `"ب"`); frontend displays `"+¡GÇ¼"`.
- After fix, entry `"1": "ب"` → frontend displays `"ب"` correctly when a Urdu-capable font is loaded.
- Confidence scores and model weights are unaffected — only the string labels change.
- Frontend PSL result panel shows garbled text even though the model prediction index is correct.

---

## Expected Behavior

### Preservation Requirements

**Bug 1 — Unchanged Behaviors:**
- Uploaded videos (non-webcam path) are processed without any trimming.
- Keypoint extraction logic (`KeypointExtractor` with `model_complexity=1`, CLAHE preprocessing) is unchanged.
- Model inference pipeline (TCN + BiGRU, temperature scaling, TTA) is unchanged.
- Short videos that become shorter than `TRIM_FRAMES` after trimming are still processed; `resample_sequence` pads them as before.
- The `use_tta` flag behavior is unchanged.

**Bug 2 — Unchanged Behaviors:**
- Manual "Stop Recording" button stops recording immediately.
- 30-second maximum duration auto-stop fires as before.
- Thumbnail capture on stop is unchanged.
- Brief landmark loss (< `ABSENCE_THRESHOLD_MS`) does not stop recording.
- The `'review'` state overlay (re-record / submit buttons) is unchanged.

**Bug 3 — Unchanged Behaviors:**
- `AlphabetClassifier` model weights and inference logic are unchanged.
- `label_map.json` is still loaded with `encoding="utf-8"` (already correct in `psl_live_service.py`).
- Confidence scores, thresholds, and session statistics are unchanged.
- MediaPipe Hands configuration (`max_num_hands=1`, `min_detection_confidence=0.5`) is unchanged.
- Non-PSL parts of the frontend are unaffected by the font addition.

**Scope:**
All inputs that do NOT match the respective bug conditions above are completely unaffected by these fixes.

---

## Hypothesized Root Cause

### Bug 1 — WLASL Frame Trimming

1. **No trim step in the pipeline**: `extract_keypoints_from_video` reads every frame from the video file with no pre- or post-processing to remove transition frames. There is no concept of "recording source" passed through the pipeline.
2. **Frontend sends the full blob**: `submitRecording` in `page.tsx` sends the entire `recordedChunksRef.current` blob without any client-side trimming.
3. **No metadata on recording source**: The `/api/ml/sign-to-text` endpoint receives a raw video file with no flag indicating whether it came from a webcam recording or a user upload, making server-side conditional trimming require a new form field.

### Bug 2 — Manual Recording Stop Required

1. **No presence monitoring during recording**: The validation polling interval (`validationIntervalRef`) is explicitly cleared when recording starts (`startRecording` calls `clearInterval`). No equivalent monitoring runs during the `'recording'` state.
2. **No absence timer**: There is no ref or state variable tracking how long body landmarks have been absent.
3. **`stopRecording` is already correct**: The existing `stopRecording` callback correctly captures a thumbnail and stops the `MediaRecorder`. It just needs to be called automatically.

### Bug 3 — PSL Urdu Text Garbled

1. **Mojibake in `label_map.json`**: The file was likely created on a system where the default encoding was Latin-1 (Windows-1252), or the JSON was written without specifying UTF-8, causing the Urdu UTF-8 byte sequences to be stored as Latin-1 characters.
2. **No Urdu font on the frontend**: Even if the label map were fixed, the browser would fall back to a system font that may not support Urdu/Arabic script, rendering boxes or incorrect glyphs.
3. **`psl_live_service.py` already opens with `encoding="utf-8"`**: The Python side reads the file correctly, but the data in the file itself is already corrupted, so the strings loaded into `self.label_map` are already mojibake.

---

## Correctness Properties

Property 1: Bug Condition — Webcam Video Frame Trimming

_For any_ webcam-recorded video where `isBugCondition_1` holds (source is webcam and video has more than `2 × TRIM_FRAMES` frames), the fixed `sign_to_text` pipeline SHALL trim the first `TRIM_FRAMES` and last `TRIM_FRAMES` frames from the extracted keypoint sequence before running inference, so that only frames containing the actual sign gesture are processed.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Bug Condition — Auto-Stop on Sustained Absence

_For any_ recording session where `isBugCondition_2` holds (state is `'recording'`, body landmarks absent for ≥ `ABSENCE_THRESHOLD_MS`), the fixed frontend SHALL automatically invoke `stopRecording()`, producing the same state transition (`'recording'` → `'review'`) and thumbnail capture as a manual stop.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 3: Bug Condition — Correct Urdu Character Display

_For any_ PSL prediction where `isBugCondition_3` holds (label map entry contains mojibake), the fixed system SHALL return and display the correct Urdu Unicode character, rendered legibly using a font that supports Urdu/Arabic script.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 4: Preservation — Upload Path Unaffected by Trimming

_For any_ video submitted via the upload path (not webcam recording), the fixed pipeline SHALL produce exactly the same keypoint sequence and inference result as the original pipeline, with no frames trimmed.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

Property 5: Preservation — Manual and Timeout Stop Unchanged

_For any_ recording session where the user manually clicks "Stop Recording" or the 30-second timeout fires, the fixed frontend SHALL produce exactly the same behavior as the original code (immediate stop, thumbnail capture, transition to `'review'`).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

Property 6: Preservation — PSL Model Inference Unchanged

_For any_ PSL inference call, the fixed system SHALL produce the same predicted class index and confidence score as the original system; only the string label looked up from the corrected label map changes.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

---

## Fix Implementation

### Bug 1 — WLASL Frame Trimming

**File**: `emotisign-backend/app/ml/wlasl_service.py`

**Function**: `extract_keypoints_from_video` / `sign_to_text`

**Specific Changes**:

1. **Add `trim_frames` parameter to `sign_to_text`**: Accept an optional `trim_frames: int = 0` argument (default 0 preserves existing behavior for all callers).

2. **Add `trim_frames` parameter to `extract_keypoints_from_video`**: Pass it through so the trim happens immediately after extraction, before resampling.

3. **Trim the keypoint array after extraction**: After `_extract_sync` returns the `(N, 126)` array, slice it:
   ```python
   if trim_frames > 0 and keypoints.shape[0] > 2 * trim_frames:
       keypoints = keypoints[trim_frames:-trim_frames]
   ```

4. **Add `WLASL_TRIM_FRAMES` environment variable**: Read `int(os.getenv("WLASL_TRIM_FRAMES", "10"))` in `get_wlasl_service()` and store it on the service instance as `self.trim_frames`.

**File**: `emotisign-backend/app/routers/ml_router.py`

**Function**: `sign_to_text_endpoint`

5. **Add `is_webcam_recording` form field**: Accept `is_webcam_recording: bool = Form(False)` and pass `trim_frames=service.trim_frames if is_webcam_recording else 0` to `ml_service.sign_to_text(...)`.

**File**: `emotisign-frontend/src/app/translate/sign-to-text/page.tsx`

**Function**: `submitRecording`

6. **Append `is_webcam_recording=true` to the FormData**: Add `formData.append('is_webcam_recording', 'true')` in `submitRecording` (webcam path). The `uploadSelectedFile` function (upload path) does not append this field, so it defaults to `false`.

---

### Bug 2 — Auto-Stop Recording

**File**: `emotisign-frontend/src/app/translate/sign-to-text/page.tsx`

**Specific Changes**:

1. **Add `absenceStartRef`**: Add `const absenceStartRef = useRef<number | null>(null)` to track when body-landmark absence began.

2. **Add `autoStopIntervalRef`**: Add `const autoStopIntervalRef = useRef<NodeJS.Timeout | null>(null)` for the presence-monitoring interval.

3. **Add `ABSENCE_THRESHOLD_MS` constant**: `const ABSENCE_THRESHOLD_MS = 1500;` at the top of the component (or as a module-level constant).

4. **Start presence-monitoring interval in `startRecording`**: After `mediaRecorder.start()` and `setState('recording')`, start a new interval (e.g., every 200 ms) that:
   - Captures a frame from `videoRef.current`
   - Posts it to `/api/ml/validate-environment`
   - Checks `data.body?.status === 'visible'`
   - If visible: resets `absenceStartRef.current = null`
   - If not visible: sets `absenceStartRef.current = absenceStartRef.current ?? Date.now()`; if `Date.now() - absenceStartRef.current >= ABSENCE_THRESHOLD_MS`, calls `stopRecording()` and clears the interval

5. **Clear the interval in `stopRecording`**: Clear `autoStopIntervalRef.current` at the start of `stopRecording` to prevent double-stop.

6. **Clear the interval in the cleanup `useEffect`**: Add `autoStopIntervalRef` to the cleanup effect.

---

### Bug 3 — PSL Urdu Text Garbled

**File**: `emotisign-backend/app/ml/models/psl/label_map.json`

1. **Re-encode the label map**: Fix the mojibake by re-encoding each value. The corrupted strings are UTF-8 bytes decoded as Latin-1; the fix is to encode each string back to Latin-1 bytes and then decode as UTF-8:
   ```python
   fixed = {k: v.encode('latin-1').decode('utf-8') for k, v in raw.items()}
   ```
   Write the corrected JSON back with `ensure_ascii=False` and `encoding='utf-8'`.

**File**: `emotisign-frontend/src/app/translate/sign-to-text/page.tsx` (and/or the PSL live page if separate)

2. **Import Noto Nastaliq Urdu font**: Add a Google Fonts import for `Noto Nastaliq Urdu` (or `Noto Naskh Arabic`) in the relevant layout or page file. Apply `font-family: 'Noto Nastaliq Urdu', serif` to the element that renders the `urdu_text` / `predicted_label` field from PSL results.

**File**: `emotisign-backend/app/ml/psl_live_service.py`

3. **Verify `ensure_ascii=False` in any JSON serialization**: The `predict` method returns a plain dict; FastAPI's `websocket.send_json` serializes it. Confirm the WebSocket send path uses `ensure_ascii=False` (FastAPI's default JSON encoder does handle Unicode correctly, but this should be verified).

---

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate each bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

---

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate each bug BEFORE implementing the fix. Confirm or refute the root cause analysis.

#### Bug 1 — Frame Trimming

**Test Plan**: Create a synthetic keypoint sequence where the first 10 and last 10 frames contain high-motion "transition" patterns and the middle frames contain a stable sign pattern. Run inference on the full sequence vs. the trimmed sequence and compare top-1 predictions.

**Test Cases**:
1. **Full sequence test**: Pass a 90-frame sequence with noisy start/end to the unfixed `predict()` — observe degraded or incorrect top-1 prediction (will fail on unfixed code).
2. **Trimmed sequence test**: Pass the same sequence with first/last 10 frames removed — observe correct top-1 prediction.
3. **Short video test**: Pass a 15-frame sequence with `trim_frames=10` — verify trimming is skipped and padding handles it.
4. **Upload path test**: Call `sign_to_text` without `is_webcam_recording=True` — verify no trimming occurs.

**Expected Counterexamples**:
- Top-1 prediction changes (often to a wrong word) when transition frames are included.
- Possible causes: transition frames introduce zero-keypoint or high-velocity keypoints that shift the resampled sequence distribution.

#### Bug 2 — Auto-Stop

**Test Plan**: Write a unit test for the absence-detection logic by mocking `validate-environment` responses and the `stopRecording` callback. Simulate a sequence of "body visible" frames followed by "body absent" frames exceeding the threshold.

**Test Cases**:
1. **Sustained absence test**: Mock 8 consecutive "body not visible" responses at 200 ms intervals (1.6 s > 1.5 s threshold) — assert `stopRecording` is called (will fail on unfixed code).
2. **Brief absence test**: Mock 5 "body not visible" responses (1.0 s < 1.5 s threshold) followed by "body visible" — assert `stopRecording` is NOT called.
3. **Manual stop test**: Simulate manual button click during absence monitoring — assert interval is cleared and no double-stop occurs.
4. **Timeout test**: Simulate 30-second elapsed time — assert existing timeout fires and auto-stop interval is also cleared.

**Expected Counterexamples**:
- `stopRecording` is never called automatically regardless of how long the user is absent (unfixed code has no monitoring interval during recording).

#### Bug 3 — Urdu Encoding

**Test Plan**: Load the current `label_map.json` and assert that each value is a valid Urdu Unicode string (code points in the Arabic/Urdu Unicode block U+0600–U+06FF or U+FB50–U+FDFF).

**Test Cases**:
1. **Label map encoding test**: Load `label_map.json` with `encoding='utf-8'` and check that `label_map["1"]` is a single Urdu character — will fail on unfixed file.
2. **PSL predict output test**: Run `PSLLiveService.predict` on a synthetic hand-landmark frame and assert `result["data"]["urdu_text"]` contains only valid Unicode Urdu characters.
3. **Frontend render test** (manual): Open the PSL live page and verify Urdu characters render correctly with the new font.

**Expected Counterexamples**:
- `label_map["1"]` returns `"+¡GÇ¼"` instead of a Urdu character.
- Possible causes: file saved with Latin-1 encoding, or JSON written without `ensure_ascii=False`.

---

### Fix Checking

**Goal**: Verify that for all inputs where each bug condition holds, the fixed function produces the expected behavior.

#### Bug 1
```
FOR ALL keypoint_sequence WHERE isBugCondition_1(sequence) DO
  trimmed := sequence[TRIM_FRAMES : -TRIM_FRAMES]
  result  := wlasl_service.predict(trimmed)
  ASSERT result.recognized_text == expected_sign
  ASSERT result.confidence >= baseline_confidence
END FOR
```

#### Bug 2
```
FOR ALL recording_session WHERE isBugCondition_2(session) DO
  result := monitorPresence(session, ABSENCE_THRESHOLD_MS)
  ASSERT stopRecording_called == true
  ASSERT state_after == "review"
  ASSERT thumbnail_captured == true
END FOR
```

#### Bug 3
```
FOR ALL label_map_entry WHERE isBugCondition_3(entry) DO
  fixed_label := load_fixed_label_map()[entry.key]
  ASSERT isValidUrduUnicode(fixed_label) == true
  ASSERT fixed_label.encode('utf-8').decode('utf-8') == fixed_label
END FOR
```

---

### Preservation Checking

**Goal**: Verify that for all inputs where each bug condition does NOT hold, the fixed function produces the same result as the original function.

```
FOR ALL input WHERE NOT isBugCondition_N(input) DO
  ASSERT fixedFunction(input) == originalFunction(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain.
- It catches edge cases that manual unit tests might miss.
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs.

**Test Cases**:
1. **Upload path preservation**: Generate random keypoint sequences and verify `sign_to_text(..., trim_frames=0)` produces identical results to the original.
2. **Brief absence preservation**: Generate random sequences of short absences (< threshold) and verify `stopRecording` is never called.
3. **PSL confidence preservation**: Verify that after fixing the label map, the integer class index and confidence score returned by `AlphabetClassifier` are identical to the original.

---

### Unit Tests

- Test `extract_keypoints_from_video` with `trim_frames=10` on a synthetic video: assert output shape is `(original_frames - 20, 126)`.
- Test `extract_keypoints_from_video` with `trim_frames=10` on a video with ≤ 20 frames: assert no trimming occurs.
- Test absence-detection logic: mock `validate-environment` and assert `stopRecording` is called after threshold.
- Test absence-detection logic: mock brief absence and assert `stopRecording` is NOT called.
- Test `label_map.json` after fix: assert all values are valid UTF-8 Urdu strings.
- Test `PSLLiveService.predict` output: assert `urdu_text` field contains valid Unicode.

### Property-Based Tests

- Generate random keypoint sequences of length 30–200 with `trim_frames` in [5, 15]: verify trimmed length is always `original - 2 * trim_frames` when `original > 2 * trim_frames`, else unchanged.
- Generate random sequences of boolean "body visible" values: verify auto-stop fires if and only if the consecutive-false run length × interval_ms ≥ `ABSENCE_THRESHOLD_MS`.
- Generate random label map entries (valid Urdu strings): verify round-trip through `encode('latin-1').decode('utf-8')` is idempotent for already-correct strings.

### Integration Tests

- Full webcam recording flow (mocked): record → submit with `is_webcam_recording=true` → verify trimmed keypoints reach the model.
- Full upload flow: upload a video → verify no trimming, same result as before.
- PSL live WebSocket session: connect, send a synthetic hand-landmark frame, verify `urdu_text` in response is a valid Urdu character.
- Frontend font rendering: load the PSL live page and verify the Urdu font is applied to the prediction label element.
