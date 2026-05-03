# Implementation Plan: WLASL-100 Model Integration

## Overview

This implementation plan breaks down the WLASL-100 ASL recognition model integration into discrete, sequential tasks. The integration replaces the placeholder sign_to_text() function with a production-ready system that includes:
- Backend ML service with MediaPipe keypoint extraction and TCN + BiGRU model inference
- Pre-recording environment validation (background, body visibility, distance)
- REST API endpoints for video processing and validation
- Frontend recording interface with real-time validation feedback

The implementation uses **Python** for backend services and **TypeScript/React** for frontend components. Each task builds incrementally, with checkpoints to ensure stability before proceeding.

## Tasks

### Phase 1: Backend Setup and Model Infrastructure

- [x] 1. Set up model directory structure and copy model files
  - Create directory `emotisign-backend/app/ml/models/wlasl100/`
  - Copy `Model/best_model.pth` to the new directory
  - Create `vocab.json` with the 100-word WLASL vocabulary mapping (0-99 to glosses)
  - Create `temperature.json` with default temperature value of 1.0
  - Verify all files exist and are readable
  - _Requirements: 13.1, 13.2, 13.3_

- [x] 2. Update backend dependencies
  - Add PyTorch dependencies to `emotisign-backend/requirements.txt`: `torch==2.0.1`, `torchvision==0.15.2`
  - Add MediaPipe: `mediapipe==0.9.3`
  - Add OpenCV: `opencv-python==4.8.1.78`
  - Ensure numpy version: `numpy==1.24.3`
  - Document GPU setup instructions in README for CUDA 11.8
  - _Requirements: 12.1, 12.2, 12.3_

- [x] 3. Create model architecture definition
  - Create file `emotisign-backend/app/ml/wlasl_model.py`
  - Implement TCN block with temporal convolutions
  - Implement BiGRU layer for temporal dependencies
  - Implement classifier head with dropout
  - Define model class matching the checkpoint architecture: feature_dim=126, num_classes=100, d_model=192, num_layers=2, dropout=0.4
  - Add model loading method that handles state dict loading
  - _Requirements: 1.1, 1.6_

- [ ]* 3.1 Write unit tests for model architecture
  - Test model initialization with correct dimensions
  - Test forward pass with dummy input (batch_size=1, seq_len=64, features=126)
  - Test output shape is (batch_size, num_classes)
  - Verify dropout is disabled in eval mode
  - _Requirements: 1.7_

### Phase 2: ML Service Core Implementation

- [x] 4. Implement MediaPipe keypoint extraction
  - Create file `emotisign-backend/app/ml/keypoint_extractor.py`
  - Initialize MediaPipe Holistic with model_complexity=2
  - Implement CLAHE contrast normalization preprocessing
  - Implement `extract_keypoints_from_frame()` method that returns (126,) array
  - Extract left hand landmarks (21 × 3 = 63 features)
  - Extract right hand landmarks (21 × 3 = 63 features)
  - Return zero-filled features when hands are not detected
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ]* 4.1 Write unit tests for keypoint extraction
  - Test extraction with valid frame containing hands
  - Test zero-filling when hands are not detected
  - Test CLAHE preprocessing is applied
  - Test output shape is (126,)
  - Verify left and right hand features are concatenated correctly
  - _Requirements: 2.6_

- [x] 5. Implement sequence preprocessing and resampling
  - In `emotisign-backend/app/ml/ml_service.py`, implement `resample_sequence()` method
  - Use linear interpolation to resample sequences to exactly 64 frames
  - Pad sequences shorter than 64 frames by repeating the last frame
  - Normalize keypoint coordinates to [0, 1] range
  - Return numpy array of shape (64, 126)
  - _Requirements: 3.4, 3.5_

- [ ]* 5.1 Write unit tests for sequence resampling
  - Test resampling from 30 frames to 64 frames (upsampling)
  - Test resampling from 100 frames to 64 frames (downsampling)
  - Test padding when input has fewer than 64 frames
  - Verify output shape is always (64, 126)
  - _Requirements: 3.4, 3.5_

- [x] 6. Implement Test-Time Augmentation (TTA) variants
  - In `emotisign-backend/app/ml/ml_service.py`, implement `build_tta_variants()` method
  - Generate center sample (frames 0-63)
  - Generate speed-up variant (first 85% of frames resampled to 64)
  - Generate slow-down variant (last 85% of frames resampled to 64)
  - Generate horizontal mirror variant (swap left/right hands, flip x coordinates)
  - Return list of 4 numpy arrays, each of shape (64, 126)
  - _Requirements: 5.1_

- [ ]* 6.1 Write unit tests for TTA variants
  - Test that 4 variants are generated
  - Test center sample uses correct frame range
  - Test speed-up and slow-down use correct frame ranges
  - Test horizontal mirror swaps hands and flips x coordinates
  - Verify all variants have shape (64, 126)
  - _Requirements: 5.1_

- [x] 7. Implement model loading and initialization
  - In `emotisign-backend/app/ml/ml_service.py`, create `WLASLModelService` class
  - Implement `__init__()` to accept model_path, vocab_path, temperature_path, device
  - Implement `load_model()` method to load checkpoint, vocabulary, and temperature
  - Set model to evaluation mode after loading
  - Handle FileNotFoundError for missing files with descriptive error messages
  - Log successful initialization with device information
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [ ]* 7.1 Write unit tests for model loading
  - Test successful loading with valid files
  - Test FileNotFoundError when checkpoint is missing
  - Test FileNotFoundError when vocabulary is missing
  - Test model is set to eval mode
  - Test device selection (CUDA vs CPU)
  - _Requirements: 1.4, 1.5, 1.6_

- [ ] 8. Checkpoint - Verify model loading and basic inference
  - Manually test model loading with actual checkpoint file
  - Create a dummy input tensor of shape (1, 64, 126)
  - Run forward pass and verify output shape is (1, 100)
  - Ensure all tests pass, ask the user if questions arise.

### Phase 3: Video Processing and Inference Pipeline

- [x] 9. Implement video file processing
  - In `emotisign-backend/app/ml/ml_service.py`, implement `extract_keypoints_from_video()` method
  - Use OpenCV to decode video file into frames
  - Process each frame through MediaPipe keypoint extractor
  - Accumulate keypoints into a list
  - Support video formats: MP4, AVI, MOV, WEBM
  - Handle corrupted videos with VideoProcessingError
  - Return numpy array of shape (num_frames, 126)
  - _Requirements: 3.1, 3.2, 3.7, 3.10_

- [ ]* 9.1 Write unit tests for video processing
  - Test extraction from valid MP4 video
  - Test handling of corrupted video file
  - Test handling of unsupported video format
  - Verify output shape is (num_frames, 126)
  - _Requirements: 3.7, 3.10_

- [x] 10. Implement inference with TTA
  - In `emotisign-backend/app/ml/ml_service.py`, implement `predict()` method
  - Accept keypoint sequence of shape (64, 126)
  - Generate TTA variants if use_tta=True
  - Convert variants to PyTorch tensors and batch them
  - Run batch inference through model
  - Average logits across all variants
  - Apply temperature scaling to logits
  - Compute softmax probabilities
  - Return top 5 predictions with confidence scores
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ]* 10.1 Write unit tests for inference
  - Test inference with TTA enabled (4 variants)
  - Test inference with TTA disabled (single variant)
  - Test temperature scaling is applied
  - Test top 5 predictions are sorted by confidence
  - Verify output format matches requirements
  - _Requirements: 5.6, 6.1, 6.2, 6.3, 6.4_

- [x] 11. Implement main sign_to_text entry point
  - In `emotisign-backend/app/ml/ml_service.py`, implement `sign_to_text()` method
  - Accept video_path and use_tta parameters
  - Extract keypoints from video
  - Resample sequence to 64 frames
  - Run inference with TTA
  - Format output as dictionary with: recognized_text, glosses, confidence, top5_predictions, frame_count, processing_time_ms, sign_language
  - Measure and log processing time
  - _Requirements: 3.6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ]* 11.1 Write integration tests for sign_to_text
  - Test end-to-end processing with sample video
  - Test output format matches schema
  - Test processing time is within acceptable range
  - Test with videos of different lengths (5s, 15s, 30s)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 12. Implement error handling and graceful degradation
  - Add try-except blocks for GPU out-of-memory errors with CPU fallback
  - Add try-except blocks for MediaPipe extraction failures with zero-filling
  - Add try-except blocks for inference errors with descriptive messages
  - Implement `sign_to_text_with_fallback()` wrapper method
  - Log warnings for degraded performance (CPU fallback, TTA disabled)
  - _Requirements: 7.1, 7.2, 7.4, 7.5_

- [ ]* 12.1 Write unit tests for error handling
  - Test GPU OOM fallback to CPU
  - Test MediaPipe failure handling
  - Test inference error handling
  - Verify error messages are descriptive
  - _Requirements: 7.1, 7.2, 7.4_

- [ ] 13. Checkpoint - Verify end-to-end video processing
  - Test with real sample videos containing ASL signs
  - Verify predictions are reasonable (top-5 contains expected signs)
  - Verify processing time meets performance targets
  - Ensure all tests pass, ask the user if questions arise.

### Phase 4: Environment Validation Service

- [x] 14. Implement background validation
  - Create file `emotisign-backend/app/services/environment_validator.py`
  - Create `EnvironmentValidator` class
  - Implement `validate_background()` method
  - Use Canny edge detection to measure background complexity
  - Calculate edge density score (edges / total pixels)
  - Return status: "clear" (score < 0.1), "acceptable" (0.1-0.2), "cluttered" (> 0.2)
  - Return score and descriptive message
  - _Requirements: 4.2, 4.3_

- [ ]* 14.1 Write unit tests for background validation
  - Test with plain background image (should return "clear")
  - Test with moderately cluttered background (should return "acceptable")
  - Test with very cluttered background (should return "cluttered")
  - Verify score is between 0.0 and 1.0
  - _Requirements: 4.3_

- [x] 15. Implement body visibility validation
  - In `EnvironmentValidator`, implement `validate_body_visibility()` method
  - Use MediaPipe Holistic to extract pose landmarks
  - Check for left hand landmarks (all 21 points)
  - Check for right hand landmarks (all 21 points)
  - Check for arm landmarks (shoulders to wrists)
  - Check for torso landmarks (shoulders to hips)
  - Return status: "visible" (all detected), "partial" (some missing), "not_visible" (insufficient)
  - Return landmarks_detected dictionary with boolean flags
  - _Requirements: 4.4, 4.5, 4.6_

- [ ]* 15.1 Write unit tests for body visibility validation
  - Test with frame showing full body (should return "visible")
  - Test with frame showing partial body (should return "partial")
  - Test with frame showing no body (should return "not_visible")
  - Verify landmarks_detected flags are correct
  - _Requirements: 4.5, 4.6_

- [x] 16. Implement distance validation
  - In `EnvironmentValidator`, implement `validate_distance()` method
  - Extract shoulder landmarks from MediaPipe pose
  - Calculate shoulder width in pixels (Euclidean distance)
  - Return status: "optimal" (200-350px), "acceptable" (150-200 or 350-400px), "out_of_range" (< 150 or > 400px)
  - Return shoulder_width_px and descriptive message
  - _Requirements: 4.7, 4.8, 4.9_

- [ ]* 16.1 Write unit tests for distance validation
  - Test with optimal shoulder width (250px)
  - Test with acceptable shoulder width (180px, 380px)
  - Test with out-of-range shoulder width (100px, 450px)
  - Verify status matches expected ranges
  - _Requirements: 4.8, 4.9_

- [x] 17. Implement comprehensive frame validation
  - In `EnvironmentValidator`, implement `validate_frame()` method
  - Call validate_background(), validate_body_visibility(), validate_distance()
  - Aggregate results into single response
  - Determine overall status: "ready" if all criteria met, "not_ready" otherwise
  - Return dictionary with background, body, distance, and overall fields
  - _Requirements: 4.12_

- [ ]* 17.1 Write integration tests for frame validation
  - Test with ideal frame (should return "ready")
  - Test with frame failing one criterion (should return "not_ready")
  - Test with frame failing multiple criteria (should return "not_ready")
  - Verify all validation results are included in response
  - _Requirements: 4.12_

- [ ] 18. Checkpoint - Verify environment validation
  - Test with real webcam frames in various conditions
  - Verify validation feedback is accurate and helpful
  - Ensure all tests pass, ask the user if questions arise.

### Phase 5: API Endpoints

- [x] 19. Create ML router and dependency injection
  - Create file `emotisign-backend/app/routers/ml_router.py`
  - Create FastAPI router with prefix `/api/ml`
  - Implement `get_ml_service()` dependency that returns singleton WLASLModelService instance
  - Implement `get_validator()` dependency that returns singleton EnvironmentValidator instance
  - Initialize services on application startup
  - _Requirements: 8.1_

- [x] 20. Implement sign-to-text endpoint
  - In `ml_router.py`, implement `POST /api/ml/sign-to-text` endpoint
  - Accept multipart/form-data with video file
  - Accept optional query parameter `use_tta` (default: true)
  - Validate file format (MP4, AVI, MOV, WEBM)
  - Validate file size (max 50MB)
  - Save uploaded file to temporary location
  - Call `ml_service.sign_to_text()` with file path
  - Delete temporary file after processing
  - Return JSON response with prediction results
  - Handle errors with appropriate HTTP status codes (400 for invalid input, 500 for processing errors)
  - _Requirements: 8.1, 8.2, 8.3, 3.8_

- [ ]* 20.1 Write API tests for sign-to-text endpoint
  - Test successful upload and processing
  - Test with invalid video format (should return 400)
  - Test with file too large (should return 400)
  - Test with corrupted video (should return 500)
  - Verify response schema matches requirements
  - _Requirements: 8.2, 8.3, 7.3, 7.5_

- [x] 21. Implement environment validation endpoint
  - In `ml_router.py`, implement `POST /api/ml/validate-environment` endpoint
  - Accept multipart/form-data with image file (JPEG, PNG)
  - Decode image using OpenCV
  - Call `validator.validate_frame()` with decoded frame
  - Return JSON response with validation results
  - Handle errors gracefully (return partial results if possible)
  - _Requirements: 8.4, 8.5_

- [ ]* 21.1 Write API tests for validation endpoint
  - Test successful validation with valid frame
  - Test with invalid image format (should return 400)
  - Test with frame containing no body (should return "not_ready")
  - Verify response schema matches requirements
  - _Requirements: 8.5_

- [x] 22. Implement ML stats endpoint
  - In `ml_router.py`, implement `GET /api/ml/stats` endpoint
  - Track total_predictions counter in ML service
  - Track average_inference_time_ms using running average
  - Track average_confidence using running average
  - Return JSON response with statistics
  - _Requirements: 11.6_

- [x] 23. Configure CORS for frontend access
  - In `emotisign-backend/main.py`, add CORS middleware
  - Allow requests from frontend origin (http://localhost:3000 for development)
  - Allow credentials and all methods
  - _Requirements: 8.6_

- [x] 24. Register ML router in main application
  - In `emotisign-backend/main.py`, import ml_router
  - Register router with `app.include_router(ml_router)`
  - Add startup event to initialize ML service and load model
  - Add shutdown event to cleanup MediaPipe resources
  - _Requirements: 1.1, 8.1_

- [ ] 25. Checkpoint - Verify API endpoints
  - Test all endpoints using Postman or curl
  - Verify CORS headers are present
  - Verify error handling works correctly
  - Ensure all tests pass, ask the user if questions arise.

### Phase 6: Frontend Implementation

- [x] 26. Create validation state types and interfaces
  - In `emotisign-frontend/src/types/index.ts`, add ValidationState interface
  - Add ValidationResult interface for background, body, distance
  - Add PredictionResult interface for sign-to-text response
  - Export all types
  - _Requirements: 9.1_

- [x] 27. Implement webcam access and validation UI
  - In `emotisign-frontend/src/app/translate/sign-to-text/page.tsx`, create SignToTextPage component
  - Add state for validationState, isValidating, isRecording, isProcessing, result, error
  - Add refs for videoRef, mediaRecorderRef, validationIntervalRef
  - Implement `startValidation()` method to request webcam permissions
  - Display webcam feed in video element
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 28. Implement real-time validation feedback
  - Implement `sendFrameForValidation()` method to capture frame from video element
  - Convert frame to Blob and send to `/api/ml/validate-environment`
  - Parse validation response and update validationState
  - Start validation interval at 5 fps (200ms) when validation begins
  - Stop validation interval when validation ends
  - _Requirements: 9.4, 9.5, 9.11_

- [x] 29. Create validation feedback UI components
  - Display three validation indicators: Background, Body Visibility, Distance
  - Show green checkmark when criterion is met
  - Show red X or warning icon when criterion fails
  - Display specific guidance messages for each failed criterion
  - Examples: "Move back from camera", "Ensure arms are visible", "Use a plain background"
  - Update indicators in real-time as validation responses arrive
  - _Requirements: 9.5, 9.6, 9.10_

- [x] 30. Implement recording button state management
  - Disable "Start Recording" button initially
  - Enable "Start Recording" button when validationState.overall === "ready"
  - Show tooltip explaining why button is disabled when not ready
  - _Requirements: 9.7, 9.13_

- [x] 31. Implement video recording functionality
  - Implement `startRecording()` method using MediaRecorder API
  - Configure MediaRecorder with video/webm codec
  - Display recording indicator (red dot) and timer during recording
  - Implement `stopRecording()` method to finalize recording
  - Collect recorded chunks into a Blob
  - Validate file size is under 50MB
  - _Requirements: 9.8, 9.9, 9.10, 9.11_

- [x] 32. Implement video upload and processing
  - Implement `uploadVideo()` method to send video Blob to `/api/ml/sign-to-text`
  - Use FormData to send multipart/form-data request
  - Display loading indicator during upload and processing
  - Handle upload progress if possible
  - Parse response and update result state
  - Handle errors and display error messages
  - _Requirements: 9.12, 9.13, 9.16_

- [x] 33. Create results display UI
  - Display recognized_text prominently in large font
  - Display top 5 predictions in a list
  - Show confidence bars for each prediction (visual representation of confidence score)
  - Display processing metadata: frame_count, processing_time_ms
  - Add "Record Again" button to restart the process
  - _Requirements: 9.14, 9.15, 9.17_

- [x] 34. Implement error handling and user feedback
  - Display error messages in a visible alert or toast
  - Provide actionable guidance for common errors
  - Allow user to retry after errors
  - Clear error state when starting new recording
  - _Requirements: 9.16_

- [x] 35. Add loading states and transitions
  - Show loading spinner during validation initialization
  - Show loading spinner during video upload
  - Show loading spinner during processing
  - Add smooth transitions between states (validation → recording → processing → results)
  - _Requirements: 9.13_

- [ ] 36. Checkpoint - Verify frontend integration
  - Test complete workflow: validation → recording → upload → results
  - Test error scenarios (no webcam, upload failure, processing error)
  - Test UI responsiveness and user experience
  - Ensure all tests pass, ask the user if questions arise.

### Phase 7: Configuration and Environment Setup

- [ ] 37. Create model configuration file
  - Create file `emotisign-backend/app/ml/config.py`
  - Define configuration class with fields: model_path, vocab_path, temperature_path, device, confidence_threshold, use_tta, max_video_size_mb, max_video_duration_sec
  - Load configuration from environment variables with defaults
  - Validate configuration on startup
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 38. Update environment variables documentation
  - Document all ML-related environment variables in `emotisign-backend/.env.example`
  - Add: MODEL_PATH, VOCAB_PATH, TEMPERATURE_PATH, ML_DEVICE, USE_TTA, CONFIDENCE_THRESHOLD, MAX_VIDEO_SIZE_MB, MAX_VIDEO_DURATION_SEC
  - Provide example values and descriptions
  - _Requirements: 10.1, 10.2_

- [x] 39. Create health check script
  - Create file `emotisign-backend/health_check.py`
  - Implement `check_model_files()` to verify model files exist
  - Implement `check_gpu()` to verify CUDA availability
  - Implement `check_mediapipe()` to verify MediaPipe initialization
  - Run all checks and report results
  - _Requirements: 12.6_

- [x] 40. Update installation documentation
  - Update `emotisign-backend/README.md` with ML setup instructions
  - Document PyTorch installation for CUDA 11.8
  - Document MediaPipe installation
  - Document model file setup process
  - Document how to run health check script
  - Add troubleshooting section for common issues
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

### Phase 8: Testing and Verification

- [x] 41. Create test fixtures and sample data
  - Create directory `emotisign-backend/tests/fixtures/`
  - Add sample video files for testing (5s, 15s, 30s)
  - Add sample frames for validation testing
  - Add corrupted video file for error testing
  - Document fixture data in README
  - _Requirements: 3.8_

- [ ]* 42. Run all unit tests and verify coverage
  - Run pytest for all unit tests
  - Verify test coverage is above 80% for ML service and environment validator
  - Fix any failing tests
  - _Requirements: All testing requirements_

- [ ]* 43. Run integration tests
  - Test end-to-end workflow with real videos
  - Test API endpoints with real requests
  - Test frontend integration with backend
  - Verify all error scenarios are handled
  - _Requirements: All integration requirements_

- [ ] 44. Performance benchmarking
  - Measure inference time with TTA on GPU (target: <500ms)
  - Measure inference time with TTA on CPU (target: <2000ms)
  - Measure keypoint extraction rate (target: >15 fps)
  - Measure end-to-end latency for 10s video (target: <3s on GPU)
  - Document results and compare with targets
  - _Requirements: 1.8, 2.7, 5.8_

- [ ] 45. Manual testing and user acceptance
  - Test with real users performing ASL signs
  - Verify validation feedback is helpful and accurate
  - Verify predictions are reasonable for known signs
  - Collect feedback on UI/UX
  - Document any issues or improvements needed
  - _Requirements: All user-facing requirements_

- [ ] 46. Final checkpoint - Complete system verification
  - Verify all requirements are met
  - Verify all tests pass
  - Verify documentation is complete
  - Verify performance targets are met
  - Ensure all tests pass, ask the user if questions arise.

### Phase 9: Documentation and Deployment Preparation

- [x] 47. Create API documentation
  - Document all ML endpoints in `emotisign-backend/docs/api.md`
  - Include request/response schemas
  - Include example requests using curl
  - Include error codes and messages
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 48. Create deployment guide
  - Document production deployment steps
  - Document GPU setup for production
  - Document environment variable configuration
  - Document model file deployment
  - Document monitoring and logging setup
  - _Requirements: 10.1, 10.4, 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 49. Create troubleshooting guide
  - Document common issues and solutions
  - Document GPU memory errors and CPU fallback
  - Document MediaPipe initialization errors
  - Document video processing errors
  - Document performance optimization tips
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 50. Final review and handoff
  - Review all code for quality and consistency
  - Review all documentation for completeness
  - Review all tests for coverage
  - Prepare demo for stakeholders
  - Hand off to deployment team

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and provide opportunities for user feedback
- The implementation uses Python for backend (ML service, environment validator, API) and TypeScript/React for frontend
- Property-based testing is explicitly excluded as the design document states this feature is not suitable for PBT
- Testing strategy focuses on unit tests, integration tests, and snapshot tests instead
- Performance targets are specified in requirements and should be verified during benchmarking
- The workflow follows a linear progression: backend setup → ML service → environment validation → API → frontend → testing → documentation
