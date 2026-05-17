# Implementation Plan

- [x] 1. Write bug condition exploration tests
  - **Property 1: Bug Condition** - WLASL Frame Trimming / Auto-Stop / Urdu Encoding
  - **CRITICAL**: These tests MUST FAIL on unfixed code — failure confirms the bugs exist
  - **DO NOT attempt to fix the tests or the code when they fail**
  - **NOTE**: These tests encode the expected behavior — they will validate the fixes when they pass after implementation
  - **GOAL**: Surface counterexamples that demonstrate each bug exists
  - Write all three bug condition tests in `emotisign-backend/tests/test_bug_conditions.py`

  **Bug 1 — WLASL Frame Trimming (Python, pytest)**
  - Construct a synthetic `(90, 126)` keypoint array where frames 0–9 and 80–89 contain high-variance "transition" values and frames 10–79 contain a stable sign pattern
  - Assert that calling the pipeline with `trim_frames=0` returns the full 90-frame array (no trimming applied)
  - Assert that the untrimmed array differs from the trimmed array (trimmed = `seq[10:-10]`, shape `(70, 126)`)
  - Assert that a short video (≤ 20 frames) is NOT trimmed even when `trim_frames=10`
  - Run on UNFIXED code — **EXPECTED OUTCOME**: Tests PASS (they document the bug condition: no trimming occurs)
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

  **Bug 2 — Auto-Stop Recording (Python, pytest — source inspection)**
  - Read `emotisign-frontend/src/app/translate/sign-to-text/page.tsx` as a text file
  - Assert that `autoStopIntervalRef` is NOT present in the file
  - Assert that `ABSENCE_THRESHOLD_MS` is NOT present in the file
  - Assert that `absenceStartRef` is NOT present in the file
  - Run on UNFIXED code — **EXPECTED OUTCOME**: Tests PASS (confirms no auto-stop mechanism exists)
  - After the fix, these tests will FAIL (confirming the fix is in place)
  - _Requirements: 1.1, 1.2, 1.3_

  **Bug 3 — Urdu Encoding (Python, pytest)**
  - Load `emotisign-backend/app/ml/models/psl/label_map.json` with `encoding='utf-8'`
  - Assert that at least one value is NOT valid Urdu Unicode (code points outside U+0600–U+06FF and U+FB50–U+FDFF)
  - Assert that `label_map["1"]` is NOT a single valid Urdu character (currently `"+¡GÇ¼"`)
  - Assert that the mojibake values can be recovered via `v.encode('latin-1').decode('utf-8')` (validates the fix strategy)
  - Run on UNFIXED code — **EXPECTED OUTCOME**: Tests PASS (confirms mojibake is present)
  - After the fix, the encoding tests will FAIL (confirming the fix is in place)
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Write preservation tests (BEFORE implementing fixes)
  - **Property 2: Preservation** - Upload Path / Brief Absence / PSL Model Inference
  - Write simple unit tests that verify non-buggy behavior is already correct on unfixed code
  - Run tests on UNFIXED code — **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

  **Preservation 1 — Upload path unaffected by trimming (Python, pytest)**
  - Test that `resample_sequence` with a 90-frame input and `target_frames=64` produces shape `(64, 126)` — unchanged by any trim logic
  - Test that calling the trim guard with `trim_frames=0` always returns the original array unchanged, for arrays of length 10, 64, and 200
  - Test that calling the trim guard with `trim_frames=10` on a 15-frame array (≤ 2×10) returns the original array unchanged
  - Verify tests PASS on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

  **Preservation 2 — Brief absence does not stop recording (Python, pytest — source inspection)**
  - Read `page.tsx` and assert that `stopRecording` is defined as a `useCallback`
  - Assert that the existing 30-second timer logic (`elapsed >= 30`) is present in the file
  - Assert that `mediaRecorder.stop()` is called inside `stopRecording` (not elsewhere unconditionally)
  - These checks confirm the manual/timeout stop paths exist and are unchanged
  - Verify tests PASS on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

  **Preservation 3 — PSL model inference unchanged (Python, pytest)**
  - Instantiate `AlphabetClassifier(num_classes=23)` directly (no model file needed — random weights)
  - Run inference on a fixed synthetic `(1, 42)` input tensor with `torch.manual_seed(0)`
  - Record the predicted class index and confidence score
  - Assert that re-running inference on the same input with the same seed produces identical results
  - This confirms the model inference is deterministic and unaffected by label map changes
  - Verify tests PASS on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Fix Bug 1 — WLASL Starting/Ending Frame Trimming

  - [x] 3.1 Add `trim_frames` parameter to `wlasl_service.py`
    - Add `trim_frames: int = 0` parameter to `extract_keypoints_from_video`
    - After `_extract_sync` returns the `(N, 126)` array, apply: `if trim_frames > 0 and keypoints.shape[0] > 2 * trim_frames: keypoints = keypoints[trim_frames:-trim_frames]`
    - Add `trim_frames: int = 0` parameter to `sign_to_text` and pass it through to `extract_keypoints_from_video`
    - Add `WLASL_TRIM_FRAMES` env var: read `int(os.getenv("WLASL_TRIM_FRAMES", "10"))` in `get_wlasl_service()` and store as `self.trim_frames`
    - _Bug_Condition: isBugCondition_1(input) where input.source == "webcam" AND totalFrames > 2 × TRIM_FRAMES_
    - _Expected_Behavior: keypoints = keypoints[TRIM_FRAMES:-TRIM_FRAMES] before inference_
    - _Preservation: trim_frames defaults to 0; upload path callers pass 0 (no trimming)_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Add `is_webcam_recording` form field to `ml_router.py`
    - Add `is_webcam_recording: bool = Form(False)` to `sign_to_text_endpoint`
    - Pass `trim_frames=ml_service.trim_frames if is_webcam_recording else 0` to `ml_service.sign_to_text(...)`
    - _Requirements: 2.1, 2.2_

  - [x] 3.3 Append `is_webcam_recording=true` to FormData in `page.tsx`
    - In `submitRecording`, add `formData.append('is_webcam_recording', 'true')` before the fetch call
    - The `uploadSelectedFile` function must NOT append this field (defaults to `false`)
    - _Requirements: 2.1, 2.2_

  - [x] 3.4 Verify bug condition exploration test (Bug 1) now passes
    - **Property 1: Expected Behavior** - WLASL Frame Trimming
    - Re-run the SAME tests from task 1 (Bug 1 section) — do NOT write new tests
    - The trim-guard tests now confirm the fix is wired correctly end-to-end
    - **EXPECTED OUTCOME**: Tests PASS (confirms Bug 1 is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.5 Verify preservation tests still pass after Bug 1 fix
    - **Property 2: Preservation** - Upload Path Unaffected
    - Re-run the SAME preservation tests from task 2 (Preservation 1) — do NOT write new tests
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions on upload path)

- [x] 4. Fix Bug 2 — Auto-Stop Recording

  - [x] 4.1 Add refs and constant for absence detection in `page.tsx`
    - Add `const absenceStartRef = useRef<number | null>(null)` to track when body-landmark absence began
    - Add `const autoStopIntervalRef = useRef<NodeJS.Timeout | null>(null)` for the presence-monitoring interval
    - Add `const ABSENCE_THRESHOLD_MS = 1500` as a module-level constant near the other constants at the top of the component
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 4.2 Start presence-monitoring interval in `startRecording` after `mediaRecorder.start()`
    - After `mediaRecorder.start()` and `setState('recording')`, start a `setInterval` at 200 ms stored in `autoStopIntervalRef.current`
    - Each tick: capture a frame from `videoRef.current`, POST to `/api/ml/validate-environment`, check `data.body?.status === 'visible'`
    - If visible: reset `absenceStartRef.current = null`
    - If not visible: set `absenceStartRef.current = absenceStartRef.current ?? Date.now()`; if `Date.now() - absenceStartRef.current >= ABSENCE_THRESHOLD_MS`, call `stopRecording()` and clear the interval
    - _Bug_Condition: isBugCondition_2(input) where state == "recording" AND bodyLandmarksDetected == false AND absenceDurationMs >= ABSENCE_THRESHOLD_MS_
    - _Expected_Behavior: stopRecording() is called, state transitions to 'review', thumbnail is captured_
    - _Preservation: brief absence (< ABSENCE_THRESHOLD_MS) does not trigger stop; manual stop and 30s timeout are unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [x] 4.3 Clear `autoStopIntervalRef` in `stopRecording` and cleanup `useEffect`
    - At the start of `stopRecording`, add `clearInterval(autoStopIntervalRef.current); autoStopIntervalRef.current = null` to prevent double-stop
    - Add `autoStopIntervalRef` to the existing cleanup `useEffect` (alongside `validationIntervalRef` and `countdownIntervalRef`)
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.4 Verify bug condition exploration test (Bug 2) now passes
    - **Property 1: Expected Behavior** - Auto-Stop Recording
    - Re-run the SAME source-inspection tests from task 1 (Bug 2 section) — do NOT write new tests
    - The tests now assert that `autoStopIntervalRef`, `ABSENCE_THRESHOLD_MS`, and `absenceStartRef` ARE present in `page.tsx`
    - **EXPECTED OUTCOME**: Tests FAIL (the absence-check assertions now fail, confirming the fix is in place)
    - Note: update the assertions in the test file to assert presence (not absence) of these identifiers after the fix
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 4.5 Verify preservation tests still pass after Bug 2 fix
    - **Property 2: Preservation** - Brief Absence and Manual Stop Unchanged
    - Re-run the SAME preservation tests from task 2 (Preservation 2) — do NOT write new tests
    - **EXPECTED OUTCOME**: Tests PASS (manual stop, timeout, and `stopRecording` callback are unchanged)

- [x] 5. Fix Bug 3 — PSL Urdu Text Garbled

  - [x] 5.1 Re-encode `label_map.json` to fix mojibake
    - Write a one-off Python script (or run inline) that loads `label_map.json` with `encoding='utf-8'`, applies `{k: v.encode('latin-1').decode('utf-8') for k, v in raw.items()}` to each value, and writes the corrected JSON back with `ensure_ascii=False` and `encoding='utf-8'`
    - Verify the corrected file: each value should be a valid Urdu Unicode character (code points U+0600–U+06FF or U+FB50–U+FDFF)
    - Example: `"1": "+¡GÇ¼"` → `"1": "ب"` (or the correct Urdu letter for that class)
    - _Bug_Condition: isBugCondition_3(input) where label_map_entry contains mojibake_
    - _Expected_Behavior: label_map values are valid Urdu Unicode strings, correctly rendered in the frontend_
    - _Preservation: AlphabetClassifier weights, inference logic, confidence scores, and thresholds are unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_

  - [x] 5.2 Add Urdu font support to the frontend
    - Import `Noto Nastaliq Urdu` from Google Fonts in the relevant layout or page file (PSL live page and/or sign-to-text page)
    - Apply `font-family: 'Noto Nastaliq Urdu', serif` (or equivalent Tailwind class) to the element that renders `urdu_text` / `predicted_label` from PSL results
    - Non-PSL parts of the frontend must remain unaffected
    - _Requirements: 2.3, 2.4_

  - [x] 5.3 Verify `psl_live_service.py` WebSocket JSON serialization handles Unicode correctly
    - Confirm that FastAPI's `websocket.send_json` correctly passes through Unicode Urdu characters without escaping them to `\uXXXX` sequences
    - No code change required if FastAPI's default encoder already handles this; document the verification result as a comment or in the PR description
    - _Requirements: 2.1, 2.2_

  - [x] 5.4 Verify bug condition exploration test (Bug 3) now passes
    - **Property 1: Expected Behavior** - Urdu Encoding
    - Re-run the SAME tests from task 1 (Bug 3 section) — do NOT write new tests
    - The encoding tests now FAIL (all values are valid Urdu Unicode, so the "at least one invalid" assertion fails — confirming the fix)
    - **EXPECTED OUTCOME**: Encoding tests FAIL (confirming Bug 3 is fixed); the `mojibake_can_be_decoded` test still passes
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 5.5 Verify preservation tests still pass after Bug 3 fix
    - **Property 2: Preservation** - PSL Model Inference Unchanged
    - Re-run the SAME preservation tests from task 2 (Preservation 3) — do NOT write new tests
    - **EXPECTED OUTCOME**: Tests PASS (class index and confidence score are identical before and after label map fix)

- [x] 6. Checkpoint — Ensure all tests pass
  - Run the full Python test suite: `pytest emotisign-backend/tests/ -v`
  - Confirm Bug 1 preservation tests pass (upload path unaffected)
  - Confirm Bug 2 preservation tests pass (manual stop and timeout unchanged)
  - Confirm Bug 3 preservation tests pass (PSL model inference unchanged)
  - Manually verify the PSL live page renders Urdu characters correctly with the new font
  - Manually verify that recording a webcam video and submitting it sends `is_webcam_recording=true` in the request (check browser DevTools Network tab)
  - Ensure all tests pass; ask the user if questions arise
