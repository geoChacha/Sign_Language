# Requirements Document: WLASL-100 Model Integration

## Introduction

This document specifies the requirements for integrating the WLASL-100 ASL recognition model into the EmotiSign application. The integration will replace the placeholder sign_to_text() function with a production-ready sign language recognition system capable of processing recorded videos with validated recording environments. The WLASL-100 model uses MediaPipe Holistic for keypoint extraction and a TCN + BiGRU architecture to recognize 100 ASL words from 64-frame sequences.

## Glossary

- **WLASL_Model**: The trained sign language recognition model (best_model.pth) using TCN + BiGRU architecture
- **MediaPipe_Extractor**: MediaPipe Holistic component that extracts hand and pose keypoints from video frames
- **Keypoint_Sequence**: A temporal sequence of 64 frames, each containing 126 hand keypoint features (63 per hand)
- **Backend_Service**: The FastAPI backend application (emotisign-backend)
- **ML_Service**: The machine learning service module (app/ml/ml_service.py) responsible for model inference
- **Environment_Validator**: The service that validates recording environment conditions before video capture
- **Frontend_Client**: The Next.js/React frontend application (emotisign-frontend)
- **Sign_To_Text_Page**: The frontend page for ASL-to-text translation (src/app/translate/sign-to-text/page.tsx)
- **Gloss**: A single ASL sign unit representing a word or concept
- **Confidence_Score**: A probability value between 0.0 and 1.0 indicating prediction certainty
- **TTA**: Test-Time Augmentation - technique using multiple input variants to improve prediction robustness
- **Temperature_Scaling**: Calibration technique that adjusts confidence scores for better probability estimates

## Requirements

### Requirement 1: Model Loading and Initialization

**User Story:** As a backend developer, I want the WLASL-100 model to load on application startup, so that inference is ready when users make requests.

#### Acceptance Criteria

1. WHEN the Backend_Service starts, THE ML_Service SHALL load the WLASL_Model from the checkpoint file (best_model.pth)
2. WHEN the Backend_Service starts, THE ML_Service SHALL load the vocabulary mapping from vocab.json
3. WHEN the Backend_Service starts, THE ML_Service SHALL load the calibrated temperature value from temperature.json if available
4. IF the checkpoint file is missing, THEN THE ML_Service SHALL log an error and raise a FileNotFoundError
5. IF the vocabulary file is missing, THEN THE ML_Service SHALL log an error and raise a FileNotFoundError
6. THE ML_Service SHALL initialize the model on the configured device (CUDA if available, otherwise CPU)
7. THE ML_Service SHALL set the model to evaluation mode after loading
8. THE ML_Service SHALL complete initialization within 10 seconds on GPU or 30 seconds on CPU

### Requirement 2: MediaPipe Keypoint Extraction

**User Story:** As a backend developer, I want to extract hand keypoints from video frames, so that the model can process them for sign recognition.

#### Acceptance Criteria

1. THE MediaPipe_Extractor SHALL extract 21 landmarks from the left hand (63 features: x, y, z coordinates)
2. THE MediaPipe_Extractor SHALL extract 21 landmarks from the right hand (63 features: x, y, z coordinates)
3. THE MediaPipe_Extractor SHALL use MediaPipe Holistic with model_complexity=2 for maximum accuracy
4. WHEN a hand is not detected in a frame, THE MediaPipe_Extractor SHALL return zero-filled features for that hand
5. THE MediaPipe_Extractor SHALL apply CLAHE contrast normalization to input frames before processing
6. THE MediaPipe_Extractor SHALL return a numpy array of shape (126,) containing concatenated left and right hand features
7. THE MediaPipe_Extractor SHALL process frames at a minimum rate of 15 frames per second on CPU

### Requirement 3: Video File Processing

**User Story:** As a user, I want to upload a video file containing ASL signs recorded with a validated environment, so that the system can translate it to text accurately.

#### Acceptance Criteria

1. WHEN a video file is uploaded, THE ML_Service SHALL decode the video into individual frames
2. WHEN frames are extracted, THE ML_Service SHALL process each frame through the MediaPipe_Extractor
3. WHEN keypoints are extracted, THE ML_Service SHALL accumulate them into a Keypoint_Sequence buffer
4. WHEN the buffer contains at least 64 frames, THE ML_Service SHALL resample the sequence to exactly 64 frames using linear interpolation
5. WHEN the sequence has fewer than 64 frames, THE ML_Service SHALL pad with the last frame to reach 64 frames
6. WHEN the Keypoint_Sequence is ready, THE ML_Service SHALL pass it to the WLASL_Model for inference
7. THE ML_Service SHALL support video formats: MP4, AVI, MOV, and WEBM
8. THE ML_Service SHALL process videos up to 30 seconds in length
9. THE ML_Service SHALL expect videos to be recorded with proper framing (validated environment with visible arms, hands, and torso)
10. IF the video file is corrupted or unreadable, THEN THE ML_Service SHALL return an error with message "Unable to process video file"

### Requirement 4: Pre-Recording Environment Validation

**User Story:** As a user, I want the system to validate my recording environment before I start recording, so that my signs are recognized accurately.

#### Acceptance Criteria

1. WHEN the user initiates environment validation, THE Environment_Validator SHALL access the webcam feed
2. WHEN webcam frames are received, THE Environment_Validator SHALL analyze background clarity using edge detection and texture analysis
3. WHEN background is analyzed, THE Environment_Validator SHALL return a validation status: "clear" (plain background), "acceptable" (minimal clutter), or "cluttered" (too busy)
4. WHEN webcam frames are received, THE Environment_Validator SHALL detect body visibility using MediaPipe Holistic pose landmarks
5. WHEN body landmarks are detected, THE Environment_Validator SHALL verify that arms (shoulders to wrists), hands (all 21 landmarks per hand), and torso (shoulders to hips) are visible
6. WHEN body visibility is checked, THE Environment_Validator SHALL return validation status: "visible" (all required landmarks detected), "partial" (some landmarks missing), or "not_visible" (insufficient landmarks)
7. WHEN webcam frames are received, THE Environment_Validator SHALL estimate user distance from camera using shoulder width in pixels
8. WHEN distance is estimated, THE Environment_Validator SHALL validate that shoulder width is between 150-400 pixels (safe distance range)
9. WHEN distance is validated, THE Environment_Validator SHALL return status: "optimal" (200-350 pixels), "acceptable" (150-200 or 350-400 pixels), or "out_of_range" (below 150 or above 400 pixels)
10. THE Environment_Validator SHALL provide real-time feedback with green checkmarks when each validation criterion is met
11. THE Environment_Validator SHALL update validation status at least 5 times per second during validation
12. THE Environment_Validator SHALL return an overall validation result: "ready" when all criteria are met (clear/acceptable background, visible body, optimal/acceptable distance), or "not_ready" otherwise
13. THE Frontend_Client SHALL only enable the recording button when validation status is "ready"

### Requirement 5: Model Inference with TTA

**User Story:** As a backend developer, I want the model to use test-time augmentation, so that predictions are more robust to variations in signing speed and orientation.

#### Acceptance Criteria

1. WHEN inference is requested with TTA enabled, THE ML_Service SHALL generate 4 sequence variants: center sample, speed-up, slow-down, and horizontal mirror
2. WHEN variants are generated, THE ML_Service SHALL process all variants through the WLASL_Model in a single batch
3. WHEN logits are obtained, THE ML_Service SHALL average the logits across all variants
4. WHEN averaged logits are computed, THE ML_Service SHALL apply temperature scaling using the calibrated temperature value
5. WHEN temperature scaling is applied, THE ML_Service SHALL compute softmax probabilities
6. THE ML_Service SHALL return the top 5 predictions sorted by Confidence_Score in descending order
7. WHERE TTA is disabled, THE ML_Service SHALL process only the center-sampled sequence
8. THE ML_Service SHALL complete inference with TTA within 500 milliseconds on GPU or 2000 milliseconds on CPU

### Requirement 6: Prediction Output Format

**User Story:** As a frontend developer, I want structured prediction results, so that I can display them to users effectively.

#### Acceptance Criteria

1. THE ML_Service SHALL return a dictionary containing the key "recognized_text" with the top predicted Gloss as a string
2. THE ML_Service SHALL return a list of "glosses" containing the top 5 predicted Gloss values
3. THE ML_Service SHALL return a "confidence" value representing the Confidence_Score of the top prediction
4. THE ML_Service SHALL return "top5_predictions" as a list of tuples containing (gloss, confidence) pairs
5. THE ML_Service SHALL return "frame_count" indicating the number of frames processed
6. THE ML_Service SHALL return "processing_time_ms" indicating the inference duration in milliseconds
7. THE ML_Service SHALL return "sign_language" with value "ASL"

### Requirement 7: Error Handling and Graceful Degradation

**User Story:** As a user, I want informative error messages when recognition fails, so that I understand what went wrong.

#### Acceptance Criteria

1. IF MediaPipe fails to extract keypoints from a frame, THEN THE ML_Service SHALL log a warning and continue with zero-filled features
2. IF the WLASL_Model raises an exception during inference, THEN THE ML_Service SHALL catch the exception and return an error response
3. IF the video file format is unsupported, THEN THE ML_Service SHALL return an error with message "Unsupported video format"
4. IF GPU memory is exhausted, THEN THE ML_Service SHALL fall back to CPU processing and log a warning
5. THE ML_Service SHALL return error responses with HTTP status code 500 for internal errors
6. THE ML_Service SHALL return error responses with HTTP status code 400 for invalid input errors

### Requirement 8: API Endpoint Integration

**User Story:** As a frontend developer, I want REST endpoints for sign recognition, so that I can integrate them into the UI.

#### Acceptance Criteria

1. THE Backend_Service SHALL expose a POST endpoint at /api/ml/sign-to-text for video file uploads
2. WHEN a POST request is received, THE Backend_Service SHALL accept multipart/form-data with a video file
3. WHEN the video is processed, THE Backend_Service SHALL return a JSON response with prediction results
4. THE Backend_Service SHALL expose a POST endpoint at /api/ml/validate-environment for environment validation
5. WHEN a validation request is received with a webcam frame, THE Backend_Service SHALL return validation results for background, body visibility, and distance
6. THE Backend_Service SHALL include CORS headers to allow requests from the Frontend_Client origin

### Requirement 9: Frontend Video Upload Integration

**User Story:** As a user, I want to record a video with environment validation and upload it for translation, so that I can get accurate ASL-to-text results.

#### Acceptance Criteria

1. THE Sign_To_Text_Page SHALL display a "Start Validation" button to begin environment checking
2. WHEN the validation button is clicked, THE Sign_To_Text_Page SHALL request webcam permissions from the browser
3. WHEN permissions are granted, THE Sign_To_Text_Page SHALL display the webcam feed with validation overlay
4. WHEN webcam is active, THE Sign_To_Text_Page SHALL send frames to /api/ml/validate-environment at 5 frames per second
5. WHEN validation responses are received, THE Sign_To_Text_Page SHALL display real-time feedback with green checkmarks for: background clarity, body visibility, and distance
6. WHEN validation responses are received, THE Sign_To_Text_Page SHALL display specific guidance messages for failed criteria (e.g., "Move back from camera", "Ensure arms are visible", "Use a plain background")
7. WHEN all validation criteria are met, THE Sign_To_Text_Page SHALL enable a "Start Recording" button
8. WHEN the recording button is clicked, THE Sign_To_Text_Page SHALL begin recording video using MediaRecorder API
9. WHEN recording is active, THE Sign_To_Text_Page SHALL display a recording indicator and timer
10. THE Sign_To_Text_Page SHALL display a "Stop Recording" button during recording
11. WHEN the stop button is clicked, THE Sign_To_Text_Page SHALL finalize the recording and validate the file size is under 50MB
12. WHEN the recording is finalized, THE Sign_To_Text_Page SHALL automatically upload the video to /api/ml/sign-to-text
13. WHEN the upload is sent, THE Sign_To_Text_Page SHALL display a loading indicator
14. WHEN the response is received, THE Sign_To_Text_Page SHALL display the recognized_text prominently
15. WHEN the response is received, THE Sign_To_Text_Page SHALL display the top 5 predictions with confidence bars
16. IF an error occurs, THEN THE Sign_To_Text_Page SHALL display the error message to the user
17. THE Sign_To_Text_Page SHALL provide an option to re-record if the user is not satisfied with the result

### Requirement 10: Model Configuration Management

**User Story:** As a backend developer, I want centralized configuration for model parameters, so that I can adjust settings without modifying code.

#### Acceptance Criteria

1. THE Backend_Service SHALL read model configuration from a config file or environment variables
2. THE Backend_Service SHALL support configuration of: model_path, vocab_path, device (cuda/cpu), confidence_threshold, and use_tta
3. WHEN configuration values are missing, THE Backend_Service SHALL use default values: confidence_threshold=0.25, use_tta=true
4. THE Backend_Service SHALL validate that model_path and vocab_path point to existing files on startup
5. IF configuration validation fails, THEN THE Backend_Service SHALL log an error and prevent startup

### Requirement 11: Performance Monitoring and Logging

**User Story:** As a system administrator, I want detailed logs of model performance, so that I can monitor and optimize the system.

#### Acceptance Criteria

1. THE ML_Service SHALL log the inference time for each prediction
2. THE ML_Service SHALL log the confidence score of the top prediction
3. THE ML_Service SHALL log the number of frames processed per request
4. THE ML_Service SHALL log warnings when inference time exceeds 1000 milliseconds
5. THE ML_Service SHALL log errors with full stack traces when exceptions occur
6. THE Backend_Service SHALL expose a /api/ml/stats endpoint returning: total_predictions, average_inference_time_ms, and average_confidence
7. THE Backend_Service SHALL reset statistics counters on application restart

### Requirement 12: Dependency Installation and Environment Setup

**User Story:** As a developer, I want clear instructions for installing model dependencies, so that I can set up the environment correctly.

#### Acceptance Criteria

1. THE Backend_Service SHALL document required Python packages: torch, torchvision, mediapipe, opencv-python, numpy
2. THE Backend_Service SHALL specify minimum versions: torch>=2.0.0, mediapipe==0.9.3, opencv-python>=4.8.0
3. THE Backend_Service SHALL provide a requirements.txt file with pinned versions
4. THE Backend_Service SHALL document the process for copying model files (best_model.pth, vocab.json) to the appropriate directory
5. THE Backend_Service SHALL document GPU setup instructions for CUDA-enabled inference
6. THE Backend_Service SHALL include a health check script to verify model loading and MediaPipe functionality

### Requirement 13: Model File Management

**User Story:** As a backend developer, I want the model files organized in a standard location, so that deployment is straightforward.

#### Acceptance Criteria

1. THE Backend_Service SHALL store model files in the directory: emotisign-backend/app/ml/models/wlasl100/
2. THE Backend_Service SHALL expect the following files in that directory: best_model.pth, vocab.json, temperature.json (optional)
3. WHEN the application starts, THE Backend_Service SHALL verify all required model files exist
4. IF model files are missing, THEN THE Backend_Service SHALL log an error with the expected file paths
5. THE Backend_Service SHALL support loading model files from an alternative path specified via environment variable MODEL_PATH
