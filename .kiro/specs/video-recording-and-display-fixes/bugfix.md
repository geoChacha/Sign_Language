# Bugfix Requirements Document

## Introduction

This document addresses three distinct bugs in the sign language translation application that affect video recording, processing, and display functionality:

1. **Starting/ending frames degrading WLASL predictions**: When users submit recorded videos for WLASL (American Sign Language) recognition, the starting and ending frames (where the user is getting into position or finishing the sign) are included in the video, which changes and degrades the prediction accuracy. These transition frames should be trimmed before processing.

2. **Manual recording stop requirement**: Currently, users must manually click the "Stop Recording" button to end video recording. For improved user experience, the system should automatically stop recording when the person moves out of frame, making the recording process more seamless.

3. **PSL Urdu text display garbled**: For PSL (Pakistan Sign Language) recognition, the predicted Urdu words are not displaying correctly - they appear as garbled text instead of proper Urdu characters, making the results unreadable for users.

These bugs impact the core functionality of the sign-to-text translation feature and affect user experience across both WLASL and PSL recognition modes.

---

## Bug Analysis

### Bug 1: Starting/Ending Frames Affecting WLASL Predictions

#### Current Behavior (Defect)

1.1 WHEN a user records a video for WLASL sign-to-text translation THEN the system includes all frames from the start of recording (including frames where the user is getting into position) in the video sent for prediction

1.2 WHEN a user records a video for WLASL sign-to-text translation THEN the system includes all frames until the end of recording (including frames where the user is finishing/relaxing after the sign) in the video sent for prediction

1.3 WHEN the video contains starting transition frames (user getting into position) THEN these frames are processed by the keypoint extraction and model inference, degrading prediction accuracy

1.4 WHEN the video contains ending transition frames (user finishing the sign) THEN these frames are processed by the keypoint extraction and model inference, degrading prediction accuracy

#### Expected Behavior (Correct)

2.1 WHEN a user records a video for WLASL sign-to-text translation THEN the system SHALL trim the starting frames (first N frames or first X seconds) before sending the video for prediction

2.2 WHEN a user records a video for WLASL sign-to-text translation THEN the system SHALL trim the ending frames (last N frames or last X seconds) before sending the video for prediction

2.3 WHEN the video has starting frames trimmed THEN only the frames containing the actual sign gesture SHALL be processed by keypoint extraction and model inference

2.4 WHEN the video has ending frames trimmed THEN only the frames containing the actual sign gesture SHALL be processed by keypoint extraction and model inference, improving prediction accuracy

#### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user uploads a pre-recorded video (not recorded through the app) THEN the system SHALL CONTINUE TO process the video without automatic trimming, as the user may have already trimmed it

3.2 WHEN the WLASL service extracts keypoints from the trimmed video THEN the system SHALL CONTINUE TO use the same keypoint extraction logic (MediaPipe Holistic with model_complexity=1)

3.3 WHEN the WLASL service performs inference on the trimmed video THEN the system SHALL CONTINUE TO use the same model inference pipeline (TCN + BiGRU with temperature scaling and TTA)

3.4 WHEN a video is too short after trimming THEN the system SHALL CONTINUE TO process it with padding as currently implemented in the resample_sequence method

---

### Bug 2: Manual Recording Stop Required

#### Current Behavior (Defect)

1.1 WHEN a user is recording a sign language video THEN the system requires the user to manually click the "Stop Recording" button to end the recording

1.2 WHEN a user moves out of frame during recording THEN the recording continues until the user manually stops it or the 30-second maximum is reached

1.3 WHEN a user completes their sign and moves away from the camera THEN the system continues recording empty frames or background-only frames

#### Expected Behavior (Correct)

2.1 WHEN a user moves out of frame during recording (body landmarks no longer detected) THEN the system SHALL automatically stop the recording after a brief delay (e.g., 1-2 seconds)

2.2 WHEN the system detects that the user has moved out of frame THEN the system SHALL trigger the same recording stop logic as the manual "Stop Recording" button

2.3 WHEN the automatic stop is triggered THEN the system SHALL transition to the 'review' state showing the recording complete overlay with re-record and submit options

#### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user manually clicks the "Stop Recording" button THEN the system SHALL CONTINUE TO stop recording immediately as it currently does

3.2 WHEN the recording reaches the 30-second maximum duration THEN the system SHALL CONTINUE TO automatically stop recording as it currently does

3.3 WHEN the recording is stopped (manually, automatically, or by timeout) THEN the system SHALL CONTINUE TO capture a thumbnail from the current video frame

3.4 WHEN the user is briefly occluded or landmarks are temporarily lost (e.g., turning slightly) THEN the system SHALL CONTINUE TO record without stopping, only stopping after sustained absence from frame

---

### Bug 3: PSL Recognition Urdu Text Garbled

#### Current Behavior (Defect)

1.1 WHEN the PSL live recognition service predicts an Urdu character THEN the predicted label displays as garbled text instead of proper Urdu characters

1.2 WHEN the PSL prediction result is sent to the frontend THEN the Urdu text is not properly encoded or decoded, resulting in unreadable characters

1.3 WHEN the user views PSL recognition results THEN they see garbled text instead of readable Urdu script

#### Expected Behavior (Correct)

2.1 WHEN the PSL live recognition service predicts an Urdu character THEN the system SHALL return the label with proper UTF-8 encoding for Urdu characters

2.2 WHEN the PSL prediction result is sent to the frontend THEN the Urdu text SHALL be properly encoded in the JSON response with UTF-8 encoding

2.3 WHEN the user views PSL recognition results THEN they SHALL see properly rendered Urdu characters in the correct script

2.4 WHEN the frontend displays PSL prediction results THEN it SHALL use a font that supports Urdu/Arabic script rendering

#### Unchanged Behavior (Regression Prevention)

3.1 WHEN the PSL model performs inference on hand landmarks THEN the system SHALL CONTINUE TO use the same AlphabetClassifier model and normalization logic

3.2 WHEN the PSL service loads the label map from label_map.json THEN the system SHALL CONTINUE TO load it with UTF-8 encoding

3.3 WHEN the PSL service returns prediction confidence scores THEN the system SHALL CONTINUE TO return the same confidence values and thresholds

3.4 WHEN the PSL service detects hand landmarks THEN the system SHALL CONTINUE TO use MediaPipe Hands with the same configuration (max_num_hands=1, min_detection_confidence=0.5)
