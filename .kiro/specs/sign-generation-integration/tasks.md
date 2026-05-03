# Implementation Plan: sign-generation-integration

## Overview

Replace the placeholder `text_to_sign()` stub with a real ASL keypoint pipeline. The backend loads `.npy` keypoint files at startup and returns raw `(N_frames, 75, 2)` landmark arrays through the existing API. The frontend replaces its `<img>` animation with an HTML5 Canvas renderer that draws a MediaPipe Holistic skeleton frame-by-frame.

## Tasks

- [x] 1. Update `SignUnit` schema to include `keypoints` field
  - Add `keypoints: Optional[List[List[List[float]]]] = None` to `SignUnit` in `emotisign-backend/app/schemas.py`
  - Import `Optional` and `List` are already present — verify no new imports needed
  - _Requirements: 3.3, 4.1_

- [x] 2. Implement `SignGenerator` class
  - [x] 2.1 Create `emotisign-backend/app/ml/sign_generator.py` with the `SignGenerator` class
    - Implement `__init__`: initialize `_cache: dict[str, list]`, `_vocabulary: set[str]`, `_frame_rate_ms: int` (read from `SIGN_FRAME_RATE_MS` env var, default 50)
    - Implement `load(keypoints_dir=None)`: resolve path from arg → `KEYPOINTS_DIR` env var → default `Featrure_sign_generation/keypoints_best`; glob `*.npy`; for each file call `np.load(allow_pickle=True)`, slice `[:, :, :2]`, validate shape `(N, 75, D≥2)`, convert to `.tolist()`, store in `_cache`; log errors/warnings per design error table; populate `_vocabulary`
    - Implement `get_sign(word)`: return `_cache.get(word)` or `None`
    - Implement `async text_to_sign(text, sign_language)`: tokenize (split whitespace, strip non-alpha, lowercase), look up each token, build sign dicts with `keypoints`/`frames`/`fingerspelled` fields, compute `total_duration_ms = sum(len(frames)) * _frame_rate_ms`, return full response dict matching existing contract
    - Implement `vocabulary` property returning `frozenset(self._vocabulary)`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 6.1, 6.2, 6.3, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3_

  - [ ]* 2.2 Write property test: Vocabulary filename mapping (Property 1)
    - Create `emotisign-backend/tests/test_sign_generator.py` with a Hypothesis test using `@given(st.lists(...))` that creates temp `.npy` files, calls `load()`, and asserts `vocabulary == set of filename stems`
    - **Property 1: Vocabulary Filename Mapping**
    - **Validates: Requirements 1.3**

  - [ ]* 2.3 Write property test: Tokenization strips non-alphabetic characters (Property 2)
    - Add to `test_sign_generator.py`: `@given(st.text())` test that calls `text_to_sign()` and asserts every token in `words` matches `[a-z]+`
    - **Property 2: Tokenization Strips Non-Alphabetic Characters**
    - **Validates: Requirements 2.1**

  - [ ]* 2.4 Write property test: Vocabulary word response shape (Property 3)
    - Add to `test_sign_generator.py`: `@given(st.sampled_from(list(VOCABULARY)))` test asserting `fingerspelled=False`, `keypoints` non-empty, each frame has 75 pairs of 2 floats, `frames=[]`, `gif_url=None`
    - **Property 3: Vocabulary Word Response Shape**
    - **Validates: Requirements 2.2, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2**

  - [ ]* 2.5 Write property test: Non-vocabulary word fingerspelling (Property 4)
    - Add to `test_sign_generator.py`: `@given(st.text(...).filter(lambda w: w not in VOCABULARY))` test asserting `fingerspelled=True`, `keypoints=None`, `frames` contains single-char strings, word in `fingerspelled_words`
    - **Property 4: Non-Vocabulary Word Fingerspelling**
    - **Validates: Requirements 2.3, 4.3, 4.4**

  - [ ]* 2.6 Write property test: Word order preservation (Property 5)
    - Add to `test_sign_generator.py`: `@given(st.lists(st.text(...), min_size=1, max_size=20))` test asserting sign order matches token order
    - **Property 5: Word Order Preservation**
    - **Validates: Requirements 2.4**

  - [ ]* 2.7 Write property test: Duration calculation (Property 6)
    - Add to `test_sign_generator.py`: `@given(st.lists(st.sampled_from(list(VOCABULARY)), min_size=1, max_size=10))` test asserting `total_duration_ms == sum(len(sign["keypoints"]) for sign in signs) * frame_rate_ms`
    - **Property 6: Duration Calculation**
    - **Validates: Requirements 4.5, 5.1**

  - [ ]* 2.8 Write property test: Cache idempotence (Property 9)
    - Add to `test_sign_generator.py`: `@given(st.sampled_from(list(VOCABULARY)))` test calling `get_sign(word)` twice and asserting identical return values
    - **Property 9: Cache Idempotence**
    - **Validates: Requirements 7.2, 7.3**

  - [ ]* 2.9 Write unit tests for `SignGenerator`
    - Add example-based tests to `test_sign_generator.py`: `test_load_populates_vocabulary`, `test_load_skips_corrupted_file`, `test_load_missing_directory`, `test_get_sign_returns_correct_shape`, `test_get_sign_unknown_word_returns_none`, `test_text_to_sign_empty_input`, `test_text_to_sign_fingerspelled_word_in_list`, `test_frame_rate_env_var`, `test_frame_rate_default`
    - _Requirements: 1.1, 1.2, 1.4, 2.5, 5.2, 6.1, 6.2, 7.2_

- [x] 3. Checkpoint — backend unit complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Wire `SignGenerator` into `ml_service.py` and `main.py`
  - [x] 4.1 Update `emotisign-backend/app/ml/ml_service.py`
    - Add module-level `_sign_generator: SignGenerator | None = None`
    - Add `set_sign_generator(sg: SignGenerator) -> None` function
    - Replace the existing `text_to_sign()` stub body with a thin delegation: if `_sign_generator is None` return empty result dict; otherwise `return await _sign_generator.text_to_sign(text, sign_language)`
    - _Requirements: 8.4_

  - [x] 4.2 Update `emotisign-backend/main.py` lifespan
    - In the `lifespan` async context manager, after existing `init_db()` / WLASL init, add a `try/except` block that imports `SignGenerator` and `set_sign_generator`, instantiates `SignGenerator`, calls `sg.load()`, calls `set_sign_generator(sg)`, and logs the vocabulary size
    - On exception, log a warning and continue (do not crash startup)
    - _Requirements: 1.1, 6.1, 6.2_

- [x] 5. Update frontend `SignData` type
  - Add `keypoints: number[][][] | null` field to the `SignData` interface in `emotisign-frontend/src/types/index.ts` (or wherever the type is defined — search for `SignData` if the path differs)
  - _Requirements: 3.3, 4.1_

- [x] 6. Implement `SkeletonCanvas` component
  - [x] 6.1 Create `emotisign-frontend/src/components/ui/SkeletonCanvas.tsx`
    - Define `SkeletonCanvasProps` interface with `frames: number[][][]`, `word: string`, `frameRateMs?: number` (default 50), `width?`, `height?`, `className?`
    - Declare `POSE_CONNECTIONS` and `HAND_CONNECTIONS` topology constants matching `test_two.py` (as listed in design)
    - Implement `drawFrame(ctx, frame, width, height)`: clear canvas with `#1a1a2e` background; split frame into body (0–32), leftHand (33–53), rightHand (54–74); implement `isValid(lm)` (both coords non-zero); implement `toPixel(lm)` scaling by canvas dimensions; draw body connections (first 12 entries cyan `#00FFFF`, rest magenta `#FF00FF`); draw left-hand connections red `#FF4444`; draw right-hand connections lime `#00FF00`; skip any connection where either landmark is invalid
    - Implement animation loop using `requestAnimationFrame` with timestamp-based frame advancement at `frameRateMs` intervals; store frame index in a `useRef` (not state) to avoid re-renders; cancel RAF on cleanup
    - Handle edge cases: null canvas context (log warning), empty frames array (no loop started)
    - _Requirements: 3.5, 3.6, 3.7, 5.3_

  - [ ]* 6.2 Write property test: Canvas renders all valid connections (Property 7)
    - Create `emotisign-frontend/src/components/ui/__tests__/SkeletonCanvas.test.ts`
    - Use `fast-check`: `fc.property(fc.array(fc.tuple(fc.float({min:0,max:1}), fc.float({min:0,max:1})), {minLength:75, maxLength:75}), ...)` — mock `CanvasRenderingContext2D`, call `drawFrame`, assert `lineTo`/`moveTo` called for every valid connection and NOT called for any connection involving a `(0,0)` landmark
    - **Property 7: Canvas Renders All Valid Connections**
    - **Validates: Requirements 3.5, 3.6**

  - [ ]* 6.3 Write property test: Coordinate scaling (Property 8)
    - Add to `SkeletonCanvas.test.ts`: `fc.property(fc.float({min:0,max:1}), fc.float({min:0,max:1}), fc.integer({min:100,max:1000}), fc.integer({min:100,max:1000}), ...)` — assert pixel coords equal `(x*W, y*H)`
    - **Property 8: Coordinate Scaling**
    - **Validates: Requirements 3.7**

- [x] 7. Update `text-to-sign/page.tsx` to use `SkeletonCanvas`
  - In `emotisign-frontend/src/app/translate/text-to-sign/page.tsx`:
    - Import `SkeletonCanvas` from `@/components/ui/SkeletonCanvas`
    - Replace `currentFrame: string | null` state with `currentKeypoints: number[][][] | null` state
    - In `translateViaRest()` (or equivalent sign-iteration logic), set `currentKeypoints` to `sign.keypoints` for vocabulary words (`sign.fingerspelled === false`) and `null` for fingerspelled words
    - Replace the `<AnimatePresence>` / `<motion.img>` animation block with a conditional: render `<SkeletonCanvas frames={currentKeypoints} word={currentWord} frameRateMs={50} width={400} height={400} />` when `currentKeypoints` is non-null, otherwise render the existing placeholder/loading state
    - Update the "Sign Preview" thumbnail strip: for each sign with non-null `keypoints`, render a small canvas showing the first frame instead of an `<img>` tag
    - _Requirements: 3.5, 3.6, 3.7, 5.3_

- [x] 8. Export `SkeletonCanvas` from the UI component index
  - Add `export { default as SkeletonCanvas } from './SkeletonCanvas'` to `emotisign-frontend/src/components/ui/index.ts`
  - _Requirements: 3.5_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis (backend) and fast-check (frontend); install with `pip install hypothesis` and `npm install --save-dev fast-check` if not already present
- The `POSE_CONNECTIONS` first-12 / rest split for cyan vs magenta coloring mirrors the `test_two.py` reference implementation
- Backend property tests should use a shared fixture that calls `SignGenerator.load()` once against the real `keypoints_best/` directory to avoid repeated disk I/O across test runs
- Frontend canvas tests mock `CanvasRenderingContext2D` (e.g., via `jest-canvas-mock` or manual spies) — no real DOM required
