# Requirements Document

## Introduction

This feature integrates the pre-recorded ASL keypoint data from `Featrure_sign_generation/keypoints_best/` into the EmotiSign application, replacing the placeholder `text_to_sign()` implementation with a real skeleton-animation-based sign generation pipeline.

The system tokenizes input text, looks up each word in a 100-word ASL vocabulary backed by `.npy` MediaPipe Holistic keypoint files, renders each keypoint frame as a PNG skeleton image, and streams the resulting base64-encoded frames back through the existing `POST /api/translate/text-to-sign` endpoint. Words outside the vocabulary fall back to the existing fingerspelling behavior. The frontend displays the frames as a smooth skeleton animation without any changes to the API contract.

## Glossary

- **Sign_Generator**: The backend component responsible for loading keypoint files, serializing keypoint data, and assembling sign data for the API response.
- **Keypoint_Renderer**: The frontend canvas-based component that draws one frame of a sign by iterating over landmark coordinates and drawing skeleton connections.
- **Vocabulary**: The set of 100 ASL words for which `.npy` keypoint files exist in `Featrure_sign_generation/keypoints_best/`.
- **Keypoint_File**: A NumPy `.npy` file of shape `(N_frames, 75, 2+)` storing normalized MediaPipe Holistic landmark coordinates (33 body + 21 left hand + 21 right hand).
- **Skeleton_Frame**: One frame of a sign animation, represented as an array of 75 `[x, y]` coordinate pairs (normalized 0.0–1.0), rendered onto a canvas by the Frontend_Animator.
- **Fingerspelling_Fallback**: The existing character-by-character letter representation used for words not present in the Vocabulary.
- **Translation_API**: The existing `POST /api/translate/text-to-sign` FastAPI endpoint.
- **Frontend_Animator**: The Next.js page at `/translate/text-to-sign` that receives keypoint arrays and renders them frame-by-frame onto an HTML5 Canvas element.
- **Frame_Rate**: The playback speed of the skeleton animation, measured in milliseconds per frame (default 50 ms).

---

## Requirements

### Requirement 1: Vocabulary Loading

**User Story:** As a backend developer, I want the Sign_Generator to load all available `.npy` keypoint files at application startup, so that translation requests are served without per-request file I/O overhead.

#### Acceptance Criteria

1. WHEN the EmotiSign backend application starts, THE Sign_Generator SHALL load all `.npy` files from the configured keypoints directory into memory.
2. WHEN a `.npy` file cannot be loaded due to a file system error, THE Sign_Generator SHALL log a warning identifying the file and continue loading the remaining files.
3. THE Sign_Generator SHALL expose the loaded Vocabulary as a set of lowercase word strings derived from the filenames (without the `.npy` extension).
4. WHEN the keypoints directory does not exist or contains zero `.npy` files, THE Sign_Generator SHALL log an error and fall back to the Fingerspelling_Fallback for all words.

---

### Requirement 2: Text Tokenization and Word Lookup

**User Story:** As a user, I want to type a sentence and have each recognizable word shown as a real ASL sign, so that I can communicate more naturally than with fingerspelling alone.

#### Acceptance Criteria

1. WHEN a translation request is received, THE Sign_Generator SHALL tokenize the input text by splitting on whitespace and removing non-alphabetic characters from each token.
2. WHEN a tokenized word (lowercased) is present in the Vocabulary, THE Sign_Generator SHALL retrieve the corresponding Keypoint_File for that word.
3. WHEN a tokenized word (lowercased) is absent from the Vocabulary, THE Sign_Generator SHALL apply the Fingerspelling_Fallback for that word and include the word in the `fingerspelled_words` list of the response.
4. THE Sign_Generator SHALL preserve the original word order from the input text in the output `signs` list.
5. WHEN the input text contains only whitespace or punctuation after tokenization, THE Sign_Generator SHALL return an empty `signs` list and an empty `words` list.

---

### Requirement 3: Keypoint Data Delivery for Canvas Rendering

**User Story:** As a user, I want to see a clear skeleton animation of each ASL sign rendered smoothly in my browser, so that I can understand the hand and body movements required without waiting for server-side image generation.

#### Acceptance Criteria

1. WHEN a translation request is received for a vocabulary word, THE Sign_Generator SHALL return the raw normalized keypoint arrays for that word rather than pre-rendered PNG images.
2. THE Sign_Generator SHALL serialize each word's keypoints as a JSON array of shape `(N_frames, 75, 2)` — containing only the X and Y coordinates of each landmark — stripped of any Z or visibility channels.
3. THE Translation_API SHALL include a `keypoints` field (array of frames) alongside `word` and `fingerspelled` in each sign object for vocabulary words.
4. THE Translation_API SHALL set `frames` to an empty list `[]` for vocabulary words (keypoints are used instead), preserving backward compatibility for any client that checks `frames`.
5. THE Frontend_Animator SHALL render each keypoint frame onto an HTML5 Canvas element using the same skeleton topology as `test_two.py`: body landmarks (indices 0–32) in cyan/magenta, left-hand landmarks (indices 33–53) in red, right-hand landmarks (indices 54–74) in lime-green.
6. WHEN a landmark has both X and Y coordinates equal to zero, THE Frontend_Animator SHALL treat that landmark as absent and SHALL NOT draw connections involving it.
7. THE Frontend_Animator SHALL scale normalized coordinates (0.0–1.0) to the canvas pixel dimensions at render time.

---

### Requirement 4: API Response Format Compatibility

**User Story:** As a frontend developer, I want the text-to-sign API to return real keypoint data in a well-defined format, so that the Frontend_Animator can render the skeleton animation without breaking existing clients.

#### Acceptance Criteria

1. THE Translation_API SHALL return each sign as a JSON object containing: `word` (string), `keypoints` (array of frames for vocabulary words), `frames` (empty list `[]` for vocabulary words, character list for fingerspelled words), `gif_url` (null for vocabulary words), and `fingerspelled` (boolean).
2. WHEN a word is served from the Vocabulary, THE Translation_API SHALL set `fingerspelled` to `false` and populate `keypoints` with the word's normalized landmark data.
3. WHEN a word uses the Fingerspelling_Fallback, THE Translation_API SHALL set `fingerspelled` to `true`, set `keypoints` to `null` or omit it, and populate `frames` with the existing single-character letter representations.
4. THE Translation_API SHALL include the `fingerspelled_words` list in the response, containing only words that used the Fingerspelling_Fallback.
5. THE Translation_API SHALL return a `total_duration_ms` value calculated as the total number of keypoint frames across all vocabulary signs multiplied by the Frame_Rate (50 ms per frame).

---

### Requirement 5: Frame Rate and Animation Playback

**User Story:** As a user, I want the sign animation to play at a natural speed, so that I can follow the movements comfortably.

#### Acceptance Criteria

1. THE Sign_Generator SHALL use a default Frame_Rate of 50 milliseconds per frame when computing `total_duration_ms`.
2. WHERE a configurable frame rate is supported, THE Sign_Generator SHALL read the Frame_Rate from an environment variable `SIGN_FRAME_RATE_MS` and fall back to 50 ms if the variable is absent or non-numeric.
3. WHEN the Frontend_Animator receives frames, THE Frontend_Animator SHALL display each frame for the duration specified by the Frame_Rate (default 100 ms per frame on the client side, matching the existing `setTimeout` interval).

---

### Requirement 6: Keypoints Data Path Configuration

**User Story:** As a developer deploying EmotiSign, I want the keypoints directory path to be configurable, so that I can place the data files wherever the deployment environment requires.

#### Acceptance Criteria

1. THE Sign_Generator SHALL read the keypoints directory path from an environment variable `KEYPOINTS_DIR`.
2. WHEN `KEYPOINTS_DIR` is not set, THE Sign_Generator SHALL default to the path `Featrure_sign_generation/keypoints_best` relative to the backend working directory.
3. IF the resolved `KEYPOINTS_DIR` path does not exist, THEN THE Sign_Generator SHALL log an error message including the resolved path and fall back to Fingerspelling_Fallback for all words.

---

### Requirement 7: Performance — In-Memory Keypoint Cache

**User Story:** As a user, I want translation responses to arrive quickly, so that the application feels responsive.

#### Acceptance Criteria

1. WHEN a translation request contains a single vocabulary word, THE Translation_API SHALL return the complete response within 500 milliseconds on a standard server CPU (no rendering overhead since keypoints are returned raw).
2. THE Sign_Generator SHALL cache the loaded and serialized keypoint arrays in process memory after the first load of each word, so that repeated requests for the same word do not re-read the `.npy` file from disk.
3. WHEN the in-memory cache is populated for a word, THE Sign_Generator SHALL serve subsequent requests for that word from the cache without any disk I/O.
4. THE in-memory cache SHALL be scoped to the process lifetime (no persistence between restarts), consistent with the 100-word fixed vocabulary.

---

### Requirement 8: Error Handling

**User Story:** As a user, I want the application to handle unexpected errors gracefully, so that a single bad word does not break the entire translation.

#### Acceptance Criteria

1. IF a Keypoint_File for a vocabulary word is corrupted or has an unexpected shape, THEN THE Sign_Generator SHALL log an error identifying the word and apply the Fingerspelling_Fallback for that word.
2. IF the Keypoint_Renderer raises an exception while rendering a frame, THEN THE Sign_Generator SHALL skip that frame, log a warning, and continue rendering the remaining frames.
3. IF all frames for a vocabulary word fail to render, THEN THE Sign_Generator SHALL apply the Fingerspelling_Fallback for that word and add it to `fingerspelled_words`.
4. WHEN an unhandled exception occurs in the Translation_API endpoint, THE Translation_API SHALL return an HTTP 500 response with a descriptive error message and SHALL NOT expose internal stack traces to the client.
