# Requirements Document

## Introduction

This document defines requirements for a Pakistan Sign Language (PSL) alphabet recognition system that classifies 37 Urdu alphabets using static right-hand poses. The system replaces the existing 5-word-sign demo with a more practical alphabet-based approach that enables spelling any word in PSL.

The new system addresses three critical limitations of the current approach:
1. **Data scarcity**: 5,112 samples (37 alphabets) vs. 706 samples (5 words) — 7× more data
2. **Pipeline mismatch**: Uses raw right-hand coordinates directly from OpenPose JSON, eliminating the MediaPipe/OpenPose feature mismatch
3. **Simplicity**: Static hand poses (single-frame classification) vs. temporal sequences (100-frame sequences)

The system uses MediaPipe for real-time feature extraction (21 right-hand landmarks × 2 coordinates = 42 dimensions), applies the same normalization as the existing `scale.scalePoints` function, and classifies each frame independently using a neural network trained on the rightHandDataset.

The requirements address seven areas:
1. Dataset loading from the alphabets_dataset directory structure
2. Feature extraction from OpenPose JSON format
3. Data preprocessing and normalization
4. Model architecture for static pose classification
5. Training pipeline with proper data splits
6. Real-time webcam demo with MediaPipe integration
7. Evaluation metrics and confidence thresholding

---

## Glossary

- **System**: The PSL Alphabet Recognition System as a whole.
- **Dataset_Loader**: The component responsible for loading OpenPose JSON files from the alphabets_dataset directory structure.
- **Feature_Extractor**: The component responsible for extracting right-hand landmarks from OpenPose JSON or MediaPipe output.
- **Preprocessor**: The component responsible for normalizing right-hand coordinates using the scalePoints algorithm.
- **Classifier**: The neural network model that maps normalized 42-dimensional hand coordinates to one of 37 alphabet classes.
- **Trainer**: The component responsible for training the Classifier with proper train/validation/test splits.
- **Demo**: The real-time webcam component that displays live alphabet predictions to the user.
- **Confidence**: The softmax probability score (range 0 to 1) for the predicted class.
- **Confidence_Threshold**: The minimum probability below which the System reports "Unknown" instead of a class label.
- **OpenPose_JSON**: The JSON file format produced by OpenPose containing `hand_right_keypoints_2d` with 21 landmarks × 3 values (x, y, confidence).
- **MediaPipe_Hand**: The MediaPipe Hands solution that detects 21 hand landmarks in real-time.
- **Normalized_Coordinates**: Right-hand (x, y) coordinates after applying scalePoints normalization (translation to origin, scaling by bounding box).
- **Static_Pose**: A single-frame hand configuration representing one alphabet sign (no temporal sequence).
- **Alphabet_Label**: One of 37 Urdu alphabet characters used as class labels.
- **INPUT_DIM**: Fixed input dimensionality of 42 (21 landmarks × 2 coordinates).

---

## Requirements

### Requirement 1: Dataset Loading from alphabets_dataset

**User Story:** As a developer, I want to load all 5,112 samples from the alphabets_dataset directory structure, so that the Classifier can be trained on the full dataset.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL recursively scan `PSL_dataset/datasets/alphabets_dataset/` and load all files matching the pattern `*_keypoints.json`.
2. WHEN loading an OpenPose JSON file, THE Feature_Extractor SHALL extract the `hand_right_keypoints_2d` array from `people[0]`.
3. THE Feature_Extractor SHALL parse `hand_right_keypoints_2d` as 21 landmarks × 3 values (x, y, confidence) and extract only the (x, y) pairs, producing a 42-dimensional vector.
4. THE Dataset_Loader SHALL derive the Alphabet_Label from the parent directory name (e.g., `PSL_dataset/datasets/alphabets_dataset/++/...` → label is `++`).
5. IF `hand_right_keypoints_2d` contains all zeros OR `people` array is empty, THEN THE Dataset_Loader SHALL skip that file and log a warning.
6. THE Dataset_Loader SHALL return a dataset containing exactly 37 unique Alphabet_Labels with a total sample count between 5,000 and 5,200.

---

### Requirement 2: Feature Extraction and Normalization

**User Story:** As a developer, I want right-hand coordinates normalized consistently between training and inference, so that the model generalizes correctly to webcam input.

#### Acceptance Criteria

1. THE Preprocessor SHALL implement the scalePoints normalization algorithm: translate hand centroid to origin, then scale by bounding box dimensions.
2. WHEN normalizing a set of 21 (x, y) landmarks, THE Preprocessor SHALL compute centroid as `(mean(x), mean(y))`, translate all points by subtracting centroid, compute bounding box as `(max(x) - min(x), max(y) - min(y))`, and divide all coordinates by the maximum of bounding box width and height.
3. IF the bounding box maximum dimension is zero (all landmarks identical), THEN THE Preprocessor SHALL return the zero vector and log a warning.
4. THE Preprocessor SHALL apply the same normalization to both training data (loaded from OpenPose JSON) and inference data (extracted from MediaPipe).
5. THE Feature_Extractor SHALL extract MediaPipe right-hand landmarks as 21 (x, y) pairs in pixel coordinates, matching the coordinate system of OpenPose JSON.

---

### Requirement 3: Model Architecture for Static Pose Classification

**User Story:** As a researcher, I want a neural network architecture suitable for static hand pose classification, so that the model achieves high accuracy on the 37-class alphabet task.

#### Acceptance Criteria

1. THE Classifier SHALL accept INPUT_DIM=42 as input (21 landmarks × 2 coordinates).
2. THE Classifier SHALL output 37 class probabilities using a softmax activation.
3. THE Classifier SHALL use a feedforward architecture with at least 2 hidden layers and dropout regularization.
4. THE Classifier SHALL be trainable on a CPU-only machine within 30 minutes for 50 epochs on the full 5,112-sample dataset.
5. THE Trainer SHALL use cross-entropy loss as the training objective.
6. THE Trainer SHALL save the trained model to `alphabet_classifier.pt` upon completion.

---

### Requirement 4: Training Pipeline with Proper Data Splits

**User Story:** As a researcher, I want the dataset split into train/validation/test sets with stratification by class, so that evaluation metrics are honest and reliable.

#### Acceptance Criteria

1. THE Trainer SHALL split the dataset into 70% training, 15% validation, and 15% test sets.
2. THE Trainer SHALL apply stratified splitting to ensure each Alphabet_Label appears proportionally in all three splits.
3. THE Trainer SHALL apply data augmentation to the training set only, using at least two of: Gaussian noise (σ ≤ 0.02), random rotation (±15°), random scaling (factor in [0.9, 1.1]), horizontal mirroring.
4. THE Trainer SHALL log epoch number, training loss, validation loss, and validation accuracy to `training_log.csv` after each epoch.
5. THE Trainer SHALL implement early stopping: IF validation loss does not improve for 10 consecutive epochs, THEN THE Trainer SHALL stop training and restore the best model weights.
6. WHEN training completes, THE Trainer SHALL print the final validation accuracy and test accuracy.

---

### Requirement 5: Evaluation Metrics and Confusion Matrix

**User Story:** As a researcher, I want detailed evaluation metrics including per-class accuracy and a confusion matrix, so that I can identify which alphabets are confused with each other.

#### Acceptance Criteria

1. THE System SHALL evaluate the trained Classifier on the held-out test set and report overall accuracy, per-class precision, per-class recall, and per-class F1-score.
2. THE System SHALL generate a 37×37 confusion matrix showing predicted vs. true labels for the test set.
3. THE System SHALL save the confusion matrix as both a text file (`confusion_matrix.txt`) and a heatmap image (`confusion_matrix.png`).
4. THE System SHALL save the evaluation report to `evaluation_report.txt` containing overall accuracy, per-class metrics, and the top 5 most-confused alphabet pairs.
5. WHEN overall test accuracy is below 0.80, THE System SHALL print a warning: "WARNING: test accuracy below 80% — consider collecting more data or tuning hyperparameters."

---

### Requirement 6: Real-Time Webcam Demo with MediaPipe Integration

**User Story:** As a user, I want to perform an alphabet sign in front of my webcam and see the recognized letter displayed on screen in real time, so that I can demonstrate the system to others.

#### Acceptance Criteria

1. THE Demo SHALL open the default webcam (device index 0) and display a live video feed in a window titled "PSL Alphabet Recognition".
2. THE Demo SHALL use MediaPipe Hands to detect the right hand in each frame and extract 21 landmarks.
3. WHEN a right hand is detected, THE Demo SHALL normalize the landmarks using the Preprocessor, classify the pose using the Classifier, and display the predicted Alphabet_Label and Confidence overlaid on the video frame.
4. WHEN the Confidence is below the Confidence_Threshold (default 0.70), THE Demo SHALL display "Unknown" instead of an Alphabet_Label.
5. WHEN no right hand is detected, THE Demo SHALL display "No hand detected" on the video frame.
6. THE Demo SHALL display a visual indicator showing the Confidence as a horizontal bar (0% to 100%) below the predicted label.
7. WHEN the user presses the 'q' key, THE Demo SHALL release the webcam and close all windows cleanly.
8. IF the webcam cannot be opened, THEN THE Demo SHALL print "Error: cannot open webcam" and exit with code 1.
9. THE Demo SHALL achieve a display frame rate of at least 20 FPS on a CPU-only machine with a modern processor.

---

### Requirement 7: Confidence Thresholding for Unknown Gestures

**User Story:** As a user, I want the system to report "Unknown" for hand poses that do not match any trained alphabet, so that I am not misled by low-confidence predictions.

#### Acceptance Criteria

1. THE System SHALL define a Confidence_Threshold of 0.70 in a shared configuration module.
2. WHEN the maximum softmax probability for a prediction is below the Confidence_Threshold, THE Classifier SHALL return "Unknown" as the predicted label.
3. THE Demo SHALL display the Confidence_Threshold as a red line on the confidence bar, so the user can see when predictions cross the threshold.
4. THE System SHALL allow the Confidence_Threshold to be adjusted via a command-line argument `--threshold` (range 0.0 to 1.0).

---

### Requirement 8: Dataset Statistics and Validation

**User Story:** As a developer, I want to verify the dataset is loaded correctly with the expected number of samples per class, so that I can detect data loading errors early.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL print a summary table showing each Alphabet_Label and its sample count after loading.
2. THE Dataset_Loader SHALL compute and print the minimum, maximum, mean, and standard deviation of samples per class.
3. IF any Alphabet_Label has fewer than 100 samples, THEN THE Dataset_Loader SHALL print a warning: "WARNING: class {label} has only {count} samples — may cause training instability."
4. THE Dataset_Loader SHALL save the dataset statistics to `dataset_statistics.txt`.

---

### Requirement 9: Unified Entry-Point Script

**User Story:** As a user, I want a single script that runs the full pipeline end-to-end, so that I can reproduce the demo from scratch with one command.

#### Acceptance Criteria

1. THE System SHALL provide a `run_pipeline.py` script that accepts a `--mode` argument with values: `load`, `train`, `evaluate`, `demo`, and `all`.
2. WHEN `--mode all` is specified, THE System SHALL execute the steps in order: dataset loading → training → evaluation → demo.
3. WHEN `--mode demo` is specified, THE System SHALL verify that `alphabet_classifier.pt` exists and, IF it does not exist, THEN THE System SHALL print "Model not found — run with --mode train first" and exit with code 1.
4. THE System SHALL print the start time, end time, and elapsed time for each pipeline step when running in `--mode all`.
5. THE System SHALL accept an optional `--threshold` argument to override the default Confidence_Threshold for demo mode.

---

### Requirement 10: MediaPipe and OpenPose Coordinate Compatibility

**User Story:** As a developer, I want to ensure MediaPipe landmarks are compatible with OpenPose training data, so that the model trained on OpenPose JSON works correctly with MediaPipe inference.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL extract MediaPipe right-hand landmarks in pixel coordinates (not normalized 0-1 coordinates).
2. THE Feature_Extractor SHALL verify that MediaPipe landmark indices match OpenPose landmark indices for the right hand (both use the same 21-landmark hand model).
3. THE System SHALL include a validation script `validate_coordinates.py` that loads one OpenPose JSON sample, extracts coordinates, applies normalization, then simulates MediaPipe extraction from the same raw coordinates and verifies the normalized outputs match within tolerance 1e-3.
4. WHEN coordinate validation fails, THE validation script SHALL print the mismatched landmark indices and exit with code 1.

