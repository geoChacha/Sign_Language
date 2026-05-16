# Requirements Document: PSL Live Webcam Inference Integration

## Introduction

This document specifies the requirements for integrating a live webcam inference system for Pakistan Sign Language (PSL) alphabet recognition into the EmotiSign application. The system will enable real-time recognition of PSL alphabet hand poses using MediaPipe Hands for landmark detection and a feedforward neural network (AlphabetClassifier) for classification. This feature will provide a dedicated PSL alphabet recognition page separate from the existing ASL sign-to-text functionality.

The integration leverages the existing PSL alphabet recognition system (located in `PSL/` directory) which uses a 42-dimensional hand coordinate input (21 landmarks × 2 coordinates) and supports dynamic class detection for all Urdu alphabet characters present in the dataset.

## Glossary

- **PSL_System**: The Pakistan Sign Language alphabet recognition system
- **AlphabetClassifier**: Feedforward neural network model (42 → 128 → 64 → num_classes) for PSL alphabet classification
- **MediaPipe_Hands**: Google's hand landmark detection solution providing 21 3D hand landmarks
- **Hand_Coordinates**: 42-dimensional feature vector containing 21 hand landmarks with x,y pixel coordinates
- **Coordinate_Normalizer**: Component that applies translation and scale invariant normalization to hand coordinates
- **WebSocket_Service**: Backend service handling real-time bidirectional communication between client and server
- **PSL_Frontend**: Dedicated Next.js page for PSL alphabet recognition
- **Confidence_Threshold**: Minimum probability value (0.70) for displaying predictions
- **Frame_Buffer**: Client-side component managing webcam frame capture and transmission
- **Backend_Service**: FastAPI-based server component handling PSL inference requests
- **Model_Checkpoint**: Serialized PyTorch model file containing trained weights and metadata
- **Label_Map**: Dictionary mapping class indices to Urdu alphabet character strings
- **RTL_Text**: Right-to-left text rendering for Urdu script display
- **Hand_Skeleton**: Visual overlay showing detected hand landmarks and connections
- **FPS_Counter**: Frames-per-second display showing real-time performance metrics

## Requirements

### Requirement 1: PSL Model Integration

**User Story:** As a backend developer, I want to integrate the PSL AlphabetClassifier model into the EmotiSign backend, so that the system can perform real-time PSL alphabet recognition.

#### Acceptance Criteria

1. THE Backend_Service SHALL load the AlphabetClassifier model from the checkpoint file at startup
2. WHEN the model checkpoint is missing, THEN THE Backend_Service SHALL log an error and disable PSL endpoints
3. THE Backend_Service SHALL load the label map JSON file containing Urdu alphabet mappings
4. THE Backend_Service SHALL initialize the Coordinate_Normalizer with translation and scale invariant normalization
5. THE Backend_Service SHALL validate that the model input dimension matches 42 (21 landmarks × 2 coordinates)
6. THE Backend_Service SHALL support dynamic class count based on the loaded model checkpoint
7. THE Backend_Service SHALL run inference on CPU (GPU optional)

### Requirement 2: MediaPipe Hand Detection Service

**User Story:** As a backend developer, I want to extract hand landmarks from webcam frames using MediaPipe Hands, so that I can generate features for the PSL classifier.

#### Acceptance Criteria

1. THE Backend_Service SHALL initialize MediaPipe_Hands with max_num_hands=1 for single-hand detection
2. WHEN a frame is received, THE Backend_Service SHALL convert it from base64 to BGR format
3. THE Backend_Service SHALL extract 21 hand landmarks with x,y pixel coordinates
4. WHEN no hand is detected, THEN THE Backend_Service SHALL return a "no hand detected" status
5. THE Backend_Service SHALL convert MediaPipe normalized coordinates [0,1] to pixel coordinates
6. THE Backend_Service SHALL apply coordinate normalization (centroid translation + bounding box scaling)
7. THE Backend_Service SHALL handle both left and right hand detections (prioritize right hand)

### Requirement 3: WebSocket Endpoint for Live PSL Inference

**User Story:** As a frontend developer, I want a WebSocket endpoint for streaming webcam frames to the backend, so that I can receive real-time PSL alphabet predictions.

#### Acceptance Criteria

1. THE Backend_Service SHALL expose a WebSocket endpoint at `/ws/translate/psl-live`
2. WHEN a client connects, THE Backend_Service SHALL send a connection acknowledgment message
3. THE Backend_Service SHALL accept frame messages with type "frame" and base64-encoded image data
4. WHEN a frame is processed, THE Backend_Service SHALL return prediction results within 100ms
5. THE Backend_Service SHALL support ping/pong messages for connection keepalive
6. THE Backend_Service SHALL handle client disconnection gracefully without crashing
7. THE Backend_Service SHALL support optional JWT authentication (guests allowed)

### Requirement 4: Real-Time Prediction Protocol

**User Story:** As a system architect, I want a well-defined message protocol for PSL live inference, so that frontend and backend can communicate reliably.

#### Acceptance Criteria

1. THE PSL_System SHALL accept client messages with fields: type, data, mime
2. WHEN type is "frame", THE PSL_System SHALL process the frame and return predictions
3. WHEN type is "config", THE PSL_System SHALL update confidence threshold settings
4. WHEN type is "stop", THE PSL_System SHALL close the session and return statistics
5. THE PSL_System SHALL send server messages with fields: event, data, timestamp
6. WHEN event is "result", THE PSL_System SHALL include: predicted_label, confidence, urdu_text
7. WHEN event is "error", THE PSL_System SHALL include a descriptive error message

### Requirement 5: Coordinate Normalization

**User Story:** As a machine learning engineer, I want hand coordinates to be normalized consistently with training data, so that the model produces accurate predictions.

#### Acceptance Criteria

1. THE Coordinate_Normalizer SHALL compute the centroid of all 21 landmarks
2. THE Coordinate_Normalizer SHALL translate all coordinates to center at origin (subtract centroid)
3. THE Coordinate_Normalizer SHALL compute bounding box dimensions (min_xy, max_xy)
4. THE Coordinate_Normalizer SHALL scale coordinates by maximum bounding box dimension
5. WHEN all landmarks are identical (degenerate case), THE Coordinate_Normalizer SHALL return a zero vector
6. THE Coordinate_Normalizer SHALL return a 42-dimensional float32 array
7. THE Coordinate_Normalizer SHALL ensure normalized values are within [-2, 2] range

### Requirement 6: Confidence-Based Prediction Filtering

**User Story:** As a user, I want to see only confident predictions, so that I am not distracted by low-quality guesses.

#### Acceptance Criteria

1. THE PSL_System SHALL compute softmax probabilities from model logits
2. THE PSL_System SHALL identify the class with maximum probability
3. WHEN confidence is below 0.70, THE PSL_System SHALL return "Low Confidence" status
4. WHEN confidence is between 0.70 and 0.84, THE PSL_System SHALL display prediction with yellow indicator
5. WHEN confidence is 0.85 or higher, THE PSL_System SHALL display prediction with green indicator
6. THE PSL_System SHALL include confidence percentage in all prediction responses
7. THE PSL_System SHALL allow confidence threshold configuration via WebSocket config message

### Requirement 7: Dedicated PSL Frontend Page

**User Story:** As a PSL user, I want a dedicated page for PSL alphabet recognition, so that I can use the system without confusion with ASL features.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL be accessible at `/translate/psl-alphabet`
2. THE PSL_Frontend SHALL display a webcam viewport with live video feed
3. THE PSL_Frontend SHALL show a "Start Recognition" button to initiate WebSocket connection
4. THE PSL_Frontend SHALL display predicted Urdu alphabet labels in real-time
5. THE PSL_Frontend SHALL render Urdu text with RTL (right-to-left) text direction
6. THE PSL_Frontend SHALL use serif font with letter-spacing for Urdu text display
7. THE PSL_Frontend SHALL include a "Stop Recognition" button to close the WebSocket connection

### Requirement 8: Hand Skeleton Visualization

**User Story:** As a user, I want to see a visual overlay of detected hand landmarks, so that I can verify my hand is being tracked correctly.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL draw hand landmarks as circles on the video feed
2. THE PSL_Frontend SHALL draw connections between landmarks to form a hand skeleton
3. WHEN no hand is detected, THE PSL_Frontend SHALL display "No hand detected" message
4. THE PSL_Frontend SHALL use green color for landmark circles
5. THE PSL_Frontend SHALL use white color for connection lines
6. THE PSL_Frontend SHALL update the skeleton overlay at the same rate as video frames
7. THE PSL_Frontend SHALL render the skeleton overlay on top of the video feed

### Requirement 9: Confidence Indicator Display

**User Story:** As a user, I want to see confidence levels with color-coded indicators, so that I can judge prediction reliability at a glance.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL display confidence as a percentage (0-100%)
2. WHEN confidence is below 70%, THE PSL_Frontend SHALL display "Low Confidence" in orange
3. WHEN confidence is 70-84%, THE PSL_Frontend SHALL display prediction label in yellow
4. WHEN confidence is 85% or higher, THE PSL_Frontend SHALL display prediction label in green
5. THE PSL_Frontend SHALL show a confidence progress bar below the predicted label
6. THE PSL_Frontend SHALL update confidence display in real-time with each prediction
7. THE PSL_Frontend SHALL display "---" when no prediction is available

### Requirement 10: FPS Counter and Performance Metrics

**User Story:** As a user, I want to see real-time performance metrics, so that I can verify the system is running smoothly.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL calculate frames per second (FPS) from webcam capture rate
2. THE PSL_Frontend SHALL display FPS in the top-right corner of the video viewport
3. THE PSL_Frontend SHALL update FPS calculation every second
4. THE PSL_Frontend SHALL display processing latency for each prediction
5. THE PSL_Frontend SHALL show total frames processed in the current session
6. THE PSL_Frontend SHALL display session duration in seconds
7. THE PSL_Frontend SHALL reset metrics when a new session starts

### Requirement 11: Webcam Access and Permissions

**User Story:** As a user, I want clear feedback about webcam access, so that I can troubleshoot permission issues.

#### Acceptance Criteria

1. WHEN the user clicks "Start Recognition", THE PSL_Frontend SHALL request webcam access
2. WHEN webcam access is denied, THE PSL_Frontend SHALL display an error message with instructions
3. WHEN webcam access is granted, THE PSL_Frontend SHALL display the live video feed
4. THE PSL_Frontend SHALL use facingMode "user" for front-facing camera
5. THE PSL_Frontend SHALL request 640x480 resolution as ideal dimensions
6. WHEN the webcam is already in use, THE PSL_Frontend SHALL display an appropriate error message
7. THE PSL_Frontend SHALL release webcam resources when the user stops recognition

### Requirement 12: WebSocket Connection Management

**User Story:** As a frontend developer, I want robust WebSocket connection handling, so that the application remains stable during network issues.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL establish WebSocket connection when user clicks "Start Recognition"
2. WHEN connection fails, THE PSL_Frontend SHALL display an error message and retry option
3. THE PSL_Frontend SHALL send ping messages every 10 seconds to maintain connection
4. WHEN connection is lost, THE PSL_Frontend SHALL attempt automatic reconnection up to 3 times
5. THE PSL_Frontend SHALL close WebSocket connection when user clicks "Stop Recognition"
6. THE PSL_Frontend SHALL handle WebSocket errors gracefully without crashing the page
7. THE PSL_Frontend SHALL display connection status (connecting, connected, disconnected)

### Requirement 13: Frame Transmission Optimization

**User Story:** As a system architect, I want efficient frame transmission, so that the system can handle real-time video without overwhelming the network.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL capture webcam frames at 30 FPS
2. THE PSL_Frontend SHALL encode frames as JPEG with 80% quality
3. THE PSL_Frontend SHALL convert frames to base64 for WebSocket transmission
4. THE PSL_Frontend SHALL send every frame to the backend (no client-side throttling)
5. THE Backend_Service SHALL process frames as they arrive without queuing
6. THE Backend_Service SHALL skip frames if processing takes longer than frame interval
7. THE PSL_System SHALL maintain frame transmission rate below 500 KB/s

### Requirement 14: Urdu Text Rendering

**User Story:** As a PSL user, I want Urdu alphabet labels displayed correctly, so that I can read the predictions naturally.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL render Urdu text with dir="rtl" attribute
2. THE PSL_Frontend SHALL use a serif font family for Urdu characters
3. THE PSL_Frontend SHALL apply letter-spacing of 0.05em for readability
4. THE PSL_Frontend SHALL display Urdu text at 3xl font size (48px)
5. THE PSL_Frontend SHALL ensure proper Unicode encoding for Urdu characters
6. THE PSL_Frontend SHALL handle Urdu text in all prediction display components
7. THE PSL_Frontend SHALL test Urdu rendering across Chrome, Firefox, and Safari browsers

### Requirement 15: Error Handling and User Feedback

**User Story:** As a user, I want clear error messages and feedback, so that I can understand and resolve issues quickly.

#### Acceptance Criteria

1. WHEN model loading fails, THE Backend_Service SHALL log the error and return HTTP 503
2. WHEN frame decoding fails, THE Backend_Service SHALL send an error event with details
3. WHEN MediaPipe detection fails, THE Backend_Service SHALL return "no hand detected" status
4. WHEN WebSocket connection fails, THE PSL_Frontend SHALL display a toast notification
5. WHEN webcam access is denied, THE PSL_Frontend SHALL show instructions to enable permissions
6. THE PSL_System SHALL log all errors with timestamps and context for debugging
7. THE PSL_Frontend SHALL provide a "Retry" button for recoverable errors

### Requirement 16: Session Statistics and Logging

**User Story:** As a system administrator, I want session statistics logged, so that I can monitor system usage and performance.

#### Acceptance Criteria

1. THE Backend_Service SHALL track total frames received per session
2. THE Backend_Service SHALL track total predictions made per session
3. THE Backend_Service SHALL calculate average confidence per session
4. THE Backend_Service SHALL log session duration when connection closes
5. THE Backend_Service SHALL log prediction distribution (count per alphabet class)
6. WHEN a session ends, THE Backend_Service SHALL send a summary message to the client
7. THE Backend_Service SHALL persist session statistics to application logs

### Requirement 17: Model Checkpoint Management

**User Story:** As a deployment engineer, I want clear model checkpoint requirements, so that I can deploy the system correctly.

#### Acceptance Criteria

1. THE PSL_System SHALL load the model checkpoint from `app/ml/models/psl/alphabet_classifier.pt`
2. THE PSL_System SHALL load the label map from `app/ml/models/psl/label_map.json`
3. THE Model_Checkpoint SHALL contain: model_state_dict, num_classes, label_map, input_dim
4. THE Label_Map SHALL map integer class indices to Urdu alphabet strings
5. WHEN checkpoint files are missing, THE Backend_Service SHALL log an error at startup
6. THE PSL_System SHALL validate checkpoint integrity (correct input_dim, valid state_dict)
7. THE PSL_System SHALL support hot-reloading of model checkpoint without server restart

### Requirement 18: Replacement of Existing PSL Service

**User Story:** As a system architect, I want to replace the existing PSL video upload service with the new live webcam inference system, so that the application uses a single, modern PSL recognition approach.

#### Acceptance Criteria

1. THE Backend_Service SHALL remove the existing PSL_Service from `app/ml/psl_service.py`
2. THE Backend_Service SHALL remove the `/api/ml/psl-sign-to-text` endpoint for video uploads
3. THE Backend_Service SHALL create a new PSL_Live_Service for WebSocket-based inference
4. THE Backend_Service SHALL remove all references to the old PSL MLP classifier (110-dim input)
5. THE Backend_Service SHALL use only the AlphabetClassifier model (42-dim input) for PSL recognition
6. THE Frontend SHALL remove PSL option from the existing sign-to-text page
7. THE Backend_Service SHALL log successful PSL live service initialization at startup

### Requirement 19: Navigation and Page Layout

**User Story:** As a user, I want intuitive navigation to the PSL alphabet page, so that I can access the feature easily.

#### Acceptance Criteria

1. THE PSL_Frontend SHALL be linked from the main translation page navigation menu
2. THE PSL_Frontend SHALL display a page title "PSL Alphabet Recognition"
3. THE PSL_Frontend SHALL include a breadcrumb navigation showing current location
4. THE PSL_Frontend SHALL display a brief description of PSL alphabet recognition
5. THE PSL_Frontend SHALL include a link back to the main sign-to-text page
6. THE PSL_Frontend SHALL use consistent styling with the rest of the EmotiSign application
7. THE PSL_Frontend SHALL be responsive and work on desktop and tablet devices

### Requirement 20: Testing and Validation

**User Story:** As a quality assurance engineer, I want comprehensive testing coverage, so that the PSL live inference system is reliable.

#### Acceptance Criteria

1. THE PSL_System SHALL include unit tests for coordinate normalization
2. THE PSL_System SHALL include unit tests for model inference with dummy inputs
3. THE PSL_System SHALL include integration tests for WebSocket message protocol
4. THE PSL_System SHALL include end-to-end tests for complete inference pipeline
5. THE PSL_System SHALL validate that normalized coordinates are within expected range
6. THE PSL_System SHALL test error handling for invalid frames and malformed messages
7. THE PSL_System SHALL include performance tests verifying <100ms inference latency

