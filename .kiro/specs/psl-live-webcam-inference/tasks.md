# Implementation Plan: PSL Live Webcam Inference Integration

## Overview

Replace the existing PSL video-upload service with a real-time WebSocket-based PSL alphabet recognition system. The backend gains a new `PSLLiveService` using MediaPipe Hands + AlphabetClassifier (42-dim input). The frontend gains a dedicated `/translate/psl-alphabet` page with live webcam capture, hand skeleton overlay, and Urdu RTL text rendering.

## Tasks

- [x] 1. Remove the existing PSL service and clean up all references
  - Delete `emotisign-backend/app/ml/psl_service.py`
  - In `emotisign-backend/app/routers/ml_router.py`: remove the `get_psl_service_dep` dependency, the `PSLModelService` import, and the entire `POST /api/ml/psl-sign-to-text` endpoint
  - In `emotisign-backend/main.py`: remove the `from app.ml.psl_service import get_psl_service` import block and the PSL service startup initialization block (the `try/except` that calls `get_psl_service()`)
  - In `emotisign-frontend/src/app/translate/sign-to-text/page.tsx`: remove the PSL option from the `<Select>` options array (`{ value: 'PSL', label: 'PSL - Pakistan Sign Language' }`), remove the PSL mode badge `<div>`, remove the PSL-specific branch in `uploadSelectedFile` and `submitRecording` (the `signLanguage === 'PSL'` conditionals that route to `psl-sign-to-text`), and remove the PSL result rendering block in the Translation Output section
  - _Requirements: 18.1, 18.2, 18.4, 18.6_

- [x] 2. Set up model files for the PSL live service
  - Copy `PSL/alphabet_classifier.pt` to `emotisign-backend/app/ml/models/psl/alphabet_classifier.pt` (create the `psl/` directory)
  - Create `emotisign-backend/app/ml/models/psl/label_map.json` by loading the checkpoint with `torch.load('PSL/alphabet_classifier.pt', map_location='cpu', weights_only=False)` and extracting the `label_map` dict; write it as `{"0": "<urdu_char>", "1": "<urdu_char>", ...}` — integer keys serialized as strings
  - Verify the checkpoint contains `model_state_dict`, `num_classes`, `label_map`, and that `fc1.weight` shape is `[128, 42]` (confirming 42-dim input)
  - _Requirements: 17.1, 17.2, 17.3, 17.4_

- [x] 3. Implement the PSL live service backend
  - Create `emotisign-backend/app/ml/psl_live_service.py`
  - Define `AlphabetClassifier` class matching `PSL/model.py` exactly: `fc1 = Linear(42, 128)`, `bn1`, `dropout1`, `fc2 = Linear(128, 64)`, `bn2`, `dropout2`, `fc3 = Linear(64, num_classes)` with ReLU activations and `dropout=0.3`
  - Implement `PSLLiveService.__init__` accepting `model_path`, `label_map_path`, `device="cpu"`, `confidence_threshold=0.70`; initialize session counters (`total_frames`, `frames_with_hands`, `predictions_made`, `total_confidence`, `session_start`)
  - Implement `PSLLiveService.load_model()`: load checkpoint with `weights_only=False`, extract `num_classes` and `label_map`, instantiate `AlphabetClassifier(num_classes=num_classes)`, load state dict, call `model.eval()`; load `label_map.json`; initialize `mediapipe.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)`
  - Implement `PSLLiveService.normalize_coordinates(coords: np.ndarray) -> np.ndarray`: mirror `PSL/preprocessor.py:normalize_hand_coords()` exactly — reshape to `(21, 2)`, subtract centroid, divide by `max(bbox_width, bbox_height)`, return flat `(42,)` float32; return zero vector for degenerate case
  - Implement `PSLLiveService.extract_hand_landmarks(frame: np.ndarray) -> Optional[np.ndarray]`: convert BGR→RGB, call `self.hands.process(rgb)`, if `multi_hand_landmarks` is non-empty extract `[[lm.x * w, lm.y * h] for lm in lms.landmark[0]]` as `(21, 2)` pixel coords, flatten to `(42,)`, call `normalize_coordinates`, return result; return `None` if no hand
  - Implement `PSLLiveService.predict(frame: np.ndarray) -> Dict`: call `extract_hand_landmarks`; if `None` return `{"event": "no_hand", ...}`; run `torch.no_grad()` inference with `F.softmax`; if `confidence < self.confidence_threshold` return `{"event": "low_confidence", ...}`; otherwise return `{"event": "result", "data": {"predicted_label": label, "urdu_text": urdu, "confidence": conf, "landmarks": [[x,y], ...]}, "timestamp": time.time()}`; update session counters
  - Implement `PSLLiveService.get_session_stats() -> Dict`: return `total_frames`, `frames_with_hands`, `predictions_made`, `average_confidence`
  - Implement module-level `get_psl_live_service()` factory that creates a **new** `PSLLiveService` instance per call (not a singleton — MediaPipe Hands is not thread-safe across concurrent WebSocket sessions); load model paths from env vars `PSL_LIVE_MODEL_PATH` and `PSL_LIVE_LABEL_MAP_PATH` with defaults `app/ml/models/psl/alphabet_classifier.pt` and `app/ml/models/psl/label_map.json`
  - _Requirements: 1.1–1.7, 2.1–2.7, 5.1–5.7, 6.1–6.7, 16.1–16.4_

  - [ ]* 3.1 Write unit tests for coordinate normalization
    - In `emotisign-backend/tests/test_psl_live_service.py`, test `normalize_coordinates` with known inputs and verify output matches `PSL/preprocessor.py:normalize_hand_coords()` for identical inputs
    - Test translation invariance: `normalize(coords + offset) == normalize(coords)` for random offsets
    - Test scale invariance: `normalize(coords * k) == normalize(coords)` for random `k > 0`
    - Test degenerate case: all-zero input returns zero vector without exception
    - Test output range: all normalized values are within `[-2, 2]`
    - **Property 1: Normalization translation invariance** — *For any* 42-dim coordinate array and any offset vector, normalizing the shifted coordinates produces the same result as normalizing the original
    - **Property 2: Normalization scale invariance** — *For any* 42-dim coordinate array and any positive scalar k, normalizing the scaled coordinates produces the same result as normalizing the original
    - **Validates: Requirements 5.1–5.7**

  - [ ]* 3.2 Write unit tests for model inference
    - Test that `AlphabetClassifier` accepts a `(1, 42)` tensor and returns `(1, num_classes)` logits
    - Test that `F.softmax` output sums to `1.0 ± 1e-5` for any random input
    - Test that predictions with `confidence < 0.70` return `"low_confidence"` event, not `"result"`
    - **Property 3: Softmax probability sum** — *For any* 42-dim input tensor, the softmax of the model output sums to 1.0 ± 1e-5
    - **Property 4: Confidence threshold enforcement** — *For any* frame where the model's max probability is below the threshold, the predict() method returns a `low_confidence` or `no_hand` event, never a `result` event
    - **Validates: Requirements 6.1–6.3, 6.7**

- [x] 4. Add the WebSocket endpoint to the ML router
  - In `emotisign-backend/app/routers/ml_router.py`, add imports: `WebSocket`, `WebSocketDisconnect` from `fastapi`; `base64`, `time`; `get_psl_live_service` from `..ml.psl_live_service`
  - Add `@router.websocket("/ws/translate/psl-live")` handler `async def psl_live_websocket(websocket: WebSocket)`
  - On connect: `await websocket.accept()`, create a fresh service via `service = get_psl_live_service()`, `await service.load_model()`, send `{"event": "connected", "message": "PSL live inference session started", "timestamp": time.time()}`
  - Main loop: `await websocket.receive_json()` → dispatch on `msg["type"]`:
    - `"frame"`: base64-decode `msg["data"]`, `cv2.imdecode` to BGR numpy array, call `await service.predict(frame)`, `await websocket.send_json(result)`; on decode failure send `{"event": "error", "message": "...", "timestamp": ...}` and continue
    - `"config"`: update `service.confidence_threshold = msg.get("confidence_threshold", 0.70)`
    - `"stop"`: call `service.get_session_stats()`, add `session_duration_seconds`, send `{"event": "session_end", "data": stats, "timestamp": ...}`, `break`
    - `"ping"`: send `{"event": "pong", "timestamp": time.time()}`
  - Catch `WebSocketDisconnect` and log session info; catch generic `Exception`, send error event, close gracefully
  - Note: the WebSocket route prefix is `/api/ml` from the router, so the full path is `/api/ml/ws/translate/psl-live` — verify this matches the frontend `WS_URL` construction or adjust the router prefix accordingly (the design specifies `/ws/translate/psl-live` without the `/api/ml` prefix; if needed, register this route directly on the FastAPI `app` in `main.py` instead of on the `ml_router`)
  - _Requirements: 3.1–3.7, 4.1–4.7, 15.1–15.2_

  - [ ]* 4.1 Write WebSocket integration tests
    - Use FastAPI `TestClient` with `websocket_connect` to test the full message protocol
    - Test: client connects and receives `connected` event
    - Test: blank/solid-color JPEG frame returns `no_hand` event
    - Test: `stop` message returns `session_end` event with valid numeric stats fields
    - Test: `ping` message returns `pong` event
    - Test: `config` message with `confidence_threshold: 0.9` is accepted without error
    - Test: corrupt base64 string returns `error` event without crashing the session
    - **Validates: Requirements 3.1–3.7, 4.1–4.7**

- [x] 5. Register PSL live service initialization in main.py
  - In `emotisign-backend/main.py` lifespan, add a startup block after the WLASL block:
    ```python
    try:
        from app.ml.psl_live_service import get_psl_live_service
        psl_live = get_psl_live_service()
        await psl_live.load_model()
        print("✅ PSL live service (AlphabetClassifier) loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: PSL live service initialization failed: {e}")
        import traceback; traceback.print_exc()
    ```
  - This validates the model file exists at startup and surfaces errors early; the per-session `get_psl_live_service()` call in the WebSocket handler still creates fresh instances
  - Update the `root()` endpoint's `websockets` dict to include `"psl_live_alphabet": "ws://127.0.0.1:8000/ws/translate/psl-live"`
  - _Requirements: 18.3, 18.7_

- [x] 6. Checkpoint — verify backend is functional
  - Ensure all existing tests still pass: `pytest emotisign-backend/tests/ -v`
  - Confirm the server starts without errors (no import errors from removed PSL service)
  - Confirm the `/api/ml/psl-sign-to-text` endpoint no longer exists (returns 404)
  - Confirm the WebSocket endpoint is reachable at the correct path
  - Ask the user if any questions arise before proceeding to frontend work.

- [x] 7. Create the WebSocket client hook
  - Create `emotisign-frontend/src/hooks/usePSLWebSocket.ts`
  - Define TypeScript interfaces: `PredictionResult` (`predicted_label`, `urdu_text`, `confidence`, `landmarks: number[][]`), `SessionStats` (`total_frames`, `frames_with_hands`, `predictions_made`, `average_confidence`, `session_duration_seconds`), `UsePSLWebSocketOptions` (`onResult`, `onNoHand`, `onLowConfidence`, `onError`, `onSessionEnd`, `onConnected`)
  - Implement `usePSLWebSocket(options)` hook with `isConnected: boolean`, `connect()`, `sendFrame(base64: string)`, `disconnect()`, `updateConfig(threshold: number)`
  - `connect()`: construct WS URL from `process.env.NEXT_PUBLIC_WS_URL` (or derive from `NEXT_PUBLIC_API_URL` by replacing `http`/`https` with `ws`/`wss`); attach `onopen`, `onmessage` (dispatch on `message.event`), `onerror`, `onclose` handlers; store in `wsRef`
  - `sendFrame()`: guard with `wsRef.current?.readyState === WebSocket.OPEN` before sending `{type: "frame", data: base64, mime: "image/jpeg"}`
  - `disconnect()`: send `{type: "stop"}` then close; set `wsRef.current = null`
  - `updateConfig()`: send `{type: "config", confidence_threshold: threshold}`
  - Implement auto-reconnect: on `onclose` if `reconnectCount < 3` and not intentionally stopped, wait 2s and call `connect()` again; expose `reconnectCount` in return value
  - Send ping every 10 seconds via `setInterval` while connected; clear on disconnect
  - _Requirements: 12.1–12.7_

- [x] 8. Create the HandSkeleton canvas component
  - Create `emotisign-frontend/src/components/HandSkeleton.tsx`
  - Props: `landmarks: number[][] | null`, `width: number`, `height: number`, `className?: string`
  - Use `useEffect` watching `landmarks` to redraw: clear canvas, draw 22 MediaPipe HAND_CONNECTIONS as white lines (`strokeStyle = 'rgba(255,255,255,0.85)'`, `lineWidth = 2`), draw 21 landmark circles as green dots (`fillStyle = '#22c55e'`, radius 4)
  - MediaPipe hand connections (hardcode the 21 pairs): thumb `[0,1],[1,2],[2,3],[3,4]`; index `[0,5],[5,6],[6,7],[7,8]`; middle `[0,9],[9,10],[10,11],[11,12]`; ring `[0,13],[13,14],[14,15],[15,16]`; pinky `[0,17],[17,18],[18,19],[19,20]`; palm `[5,9],[9,13],[13,17]`
  - When `landmarks` is `null` or empty, clear the canvas and return without drawing
  - Render as `<canvas ref={canvasRef} width={width} height={height} className={className} />`
  - _Requirements: 8.1–8.7_

- [x] 9. Create the PSL alphabet page
  - Create `emotisign-frontend/src/app/translate/psl-alphabet/page.tsx` as a `'use client'` component
  - State: `isRecognizing: boolean`, `prediction: PredictionResult | null`, `noHandDetected: boolean`, `fps: number`, `sessionStats: SessionStats | null`, `error: string | null`, `landmarks: number[][] | null`, `confidenceLevel: 'high' | 'medium' | 'low' | null`
  - Refs: `videoRef` (`HTMLVideoElement`), `canvasRef` (`HTMLCanvasElement`), `streamRef` (`MediaStream`), `frameIntervalRef` (`NodeJS.Timeout`), `fpsCounterRef` (frame count + last timestamp)
  - Webcam management: `startWebcam()` calls `navigator.mediaDevices.getUserMedia({video: {facingMode: 'user', width: {ideal: 640}, height: {ideal: 480}}, audio: false})`; on `NotAllowedError` set error "Camera permission denied. Please allow camera access in your browser settings."; on `NotReadableError` set error "Camera is already in use by another application."
  - Frame capture loop: `setInterval` at 33ms; inside, draw `videoRef` to an offscreen `<canvas>` (640×480), call `canvas.toBlob(blob => ..., 'image/jpeg', 0.8)`, convert blob to base64 via `FileReader`, call `sendFrame(base64)`; track frame count for FPS; update FPS state every second using `(frameCount / elapsed).toFixed(1)`
  - Use `usePSLWebSocket` hook with callbacks: `onResult` → set `prediction`, `landmarks`, `noHandDetected=false`, derive `confidenceLevel`; `onNoHand` → set `noHandDetected=true`, `landmarks=null`; `onLowConfidence` → set `noHandDetected=false`, `prediction` with low confidence; `onSessionEnd` → set `sessionStats`, stop recognition; `onError` → show toast
  - `startRecognition()`: call `startWebcam()`, then `connect()`, then start frame interval
  - `stopRecognition()`: clear frame interval, call `disconnect()`, stop webcam tracks, reset state
  - Layout structure:
    - Page header: `<h1>PSL Alphabet Recognition</h1>` with 🇵🇰 badge and brief description
    - Webcam viewport: `<div className="relative">` containing `<video ref={videoRef} autoPlay playsInline muted />` and `<HandSkeleton landmarks={landmarks} width={640} height={480} className="absolute inset-0 pointer-events-none" />`; FPS badge top-right; "No hand detected" overlay when `noHandDetected`
    - Prediction display: large Urdu text `<p dir="rtl" style={{fontFamily: 'serif', letterSpacing: '0.05em', fontSize: '3rem'}}>` colored by confidence (green ≥ 0.85, yellow 0.70–0.84, orange < 0.70); confidence percentage and progress bar; "---" when no prediction
    - Controls: "Start Recognition" / "Stop Recognition" button; connection status badge
    - Session stats panel (shown after session ends): total frames, frames with hands, predictions made, average confidence, duration
    - Error display with retry button
  - _Requirements: 7.1–7.7, 8.1–8.7, 9.1–9.7, 10.1–10.7, 11.1–11.7, 14.1–14.7, 15.4–15.5, 19.1–19.7_

  - [ ]* 9.1 Write unit tests for confidence color logic
    - Extract the confidence-to-color mapping into a pure function `getConfidenceLevel(confidence: number): 'high' | 'medium' | 'low'`
    - Test: `confidence < 0.70` → `'low'` (orange display)
    - Test: `confidence >= 0.70 && confidence < 0.85` → `'medium'` (yellow display)
    - Test: `confidence >= 0.85` → `'high'` (green display)
    - **Property 5: Confidence color coverage** — *For any* confidence value in `[0, 1]`, `getConfidenceLevel` returns exactly one of `'high'`, `'medium'`, or `'low'`
    - **Validates: Requirements 6.4–6.5, 9.2–9.4**

- [x] 10. Add navigation link to the PSL alphabet page
  - In `emotisign-frontend/src/app/dashboard/page.tsx`, add a new entry to the `connectionOptions` array:
    ```typescript
    {
      id: 'psl-alphabet',
      title: 'PSL Alphabet (Live)',
      description: 'Recognize Pakistan Sign Language alphabet letters in real-time using your webcam.',
      href: '/translate/psl-alphabet',
      icon: <svg ...>, // hand/alphabet icon similar to existing entries
      color: 'green',
    }
    ```
  - Verify the new card renders correctly alongside existing options using the existing card component pattern
  - _Requirements: 19.1, 19.2, 19.4_

- [x] 11. Final checkpoint — ensure all tests pass
  - Run `pytest emotisign-backend/tests/ -v` and confirm all backend tests pass
  - Verify the frontend builds without TypeScript errors: `npm run build` in `emotisign-frontend/`
  - Manually verify the PSL alphabet page is accessible at `/translate/psl-alphabet` and the dashboard card links to it
  - Confirm the old sign-to-text page no longer shows a PSL option in the language selector
  - Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- The WebSocket endpoint path must be consistent between backend and frontend — double-check whether the `ml_router` prefix (`/api/ml`) applies to WebSocket routes in FastAPI; if it does, the frontend WS URL must be `ws://.../api/ml/ws/translate/psl-live`
- `get_psl_live_service()` intentionally creates a new instance per WebSocket session (not a singleton) because `mediapipe.solutions.hands.Hands` is not thread-safe
- The `alphabet_classifier.pt` model file must be manually copied from `PSL/` to `emotisign-backend/app/ml/models/psl/` before running the backend — this is a one-time setup step
- Property tests use `pytest` with `hypothesis` library for property-based testing (already available or add `hypothesis` to `requirements.txt`)
