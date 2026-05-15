# Implementation Plan: Right-Hand Alphabet Recognition System

## Overview

This implementation plan creates a PSL alphabet recognition system for 37 Urdu alphabets using static right-hand poses. The system uses a feedforward neural network (42 → 128 → 64 → 37) trained on 5,112 samples from the rightHandDataset, with MediaPipe for real-time inference.

**Key Implementation Strategy:**
- Build modules in dependency order (config → preprocessor → dataset_loader → model → train → evaluate/demo)
- Each module includes error handling for edge cases
- Property-based tests validate universal correctness properties
- Checkpoints ensure incremental validation

**Architecture:** 8 core modules + 1 orchestrator + 1 validator + property tests

---

## Tasks

### 1. Create Configuration Module

- [x] 1.1 Create `config.py` with all system constants
  - Define dataset paths: `DATASET_ROOT`, `MODEL_PATH`, output file paths
  - Define model architecture: `INPUT_DIM=42`, `NUM_CLASSES=37`, `HIDDEN_DIM_1=128`, `HIDDEN_DIM_2=64`, `DROPOUT=0.3`
  - Define training hyperparameters: `TRAIN_SPLIT=0.70`, `VAL_SPLIT=0.15`, `TEST_SPLIT=0.15`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001`, `MAX_EPOCHS=50`, `EARLY_STOPPING_PATIENCE=10`
  - Define augmentation parameters: `AUGMENT_NOISE_STD=0.02`, `AUGMENT_ROTATION_DEG=15.0`, `AUGMENT_SCALE_RANGE=(0.9, 1.1)`
  - Define inference parameters: `CONFIDENCE_THRESHOLD=0.70`
  - Add docstring explaining this is the single source of truth for all hyperparameters
  - _Requirements: 3.1, 7.1, 9.5_

---

### 2. Implement Preprocessing Module

- [x] 2.1 Create `preprocessor.py` with scalePoints normalization
  - Implement `normalize_hand_coords(coords: np.ndarray) -> np.ndarray` function
  - Accept input as (42,) or (21, 2) array, reshape to (21, 2) if needed
  - Compute centroid as mean of all (x, y) points
  - Translate all points by subtracting centroid
  - Compute bounding box as (max_x - min_x, max_y - min_y)
  - Scale by maximum of bounding box dimensions
  - Handle edge case: if max dimension is zero (all points identical), return zero vector
  - Return normalized coordinates as (42,) flat array
  - Add docstring explaining translation and scale invariance properties
  - _Requirements: 2.1, 2.2, 2.3_

- [ ]* 2.2 Write property test for normalization invariance
  - **Property 2: Normalization Translation and Scale Invariance**
  - **Validates: Requirements 2.1, 2.2**
  - Generate random hand coordinates (21 points)
  - Apply random translation (shift by random vector)
  - Apply random uniform scaling (multiply by random positive constant)
  - Verify normalized outputs are identical within tolerance 1e-5
  - Test with 100 random inputs using Hypothesis
  - _Requirements: 2.1, 2.2_

---

### 3. Implement Dataset Loading Module

- [x] 3.1 Create `dataset_loader.py` with coordinate extraction
  - Implement `extract_right_hand_coords(json_path: Path) -> np.ndarray | None`
  - Load JSON file and parse `people[0]["hand_right_keypoints_2d"]`
  - Verify array has exactly 63 values (21 landmarks × 3)
  - Extract x, y pairs (skip confidence): `[hand_right[i:i+2] for i in range(0, 63, 3)]`
  - Flatten to (42,) array
  - Return None if: `people` array is empty, array length is not 63, or all coordinates are zero
  - Add error handling for JSON parse errors and missing keys
  - _Requirements: 1.2, 1.3, 1.5_

- [x] 3.2 Implement stratified dataset splitting
  - Implement `load_dataset(root_dir, train_split, val_split, test_split) -> tuple`
  - Use `Path(root_dir).rglob("*_keypoints.json")` to find all JSON files
  - Extract label from parent directory name: `json_path.parent.parent.name`
  - Load all valid samples using `extract_right_hand_coords()`
  - Create label encoding: map string labels to integers 0-36 (alphabetical order)
  - Apply stratified splitting using `sklearn.model_selection.train_test_split` with `stratify=y`
  - Return `(X_train, y_train), (X_val, y_val), (X_test, y_test), label_map`
  - Validate: total samples between 5,000 and 5,200, exactly 37 unique labels
  - _Requirements: 1.1, 1.4, 1.6, 4.1, 4.2_

- [x] 3.3 Add dataset statistics reporting
  - Implement `print_dataset_statistics(label_counts, output_path)`
  - Compute per-class counts, min, max, mean, standard deviation
  - Print formatted table with all statistics
  - Save to `dataset_statistics.txt`
  - Warn if any class has < 100 samples
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ]* 3.4 Write property test for coordinate extraction
  - **Property 1: Coordinate Extraction Pipeline Correctness**
  - **Validates: Requirements 1.2, 1.3, 1.4**
  - Generate random valid OpenPose JSON structures with `hand_right_keypoints_2d`
  - Verify extraction produces (42,) array
  - Verify element 2i contains x-coordinate of landmark i
  - Verify element 2i+1 contains y-coordinate of landmark i
  - Test with 100 random JSON structures using Hypothesis
  - _Requirements: 1.2, 1.3, 1.4_

- [ ]* 3.5 Write property test for stratified splitting
  - **Property 4: Stratified Dataset Splitting Preserves Class Distribution**
  - **Validates: Requirements 4.1, 4.2**
  - Generate random multi-class datasets with varying class distributions
  - Apply stratified splitting with (0.70, 0.15, 0.15) ratios
  - Verify each class appears in all three splits
  - Verify class proportions are within ±5% of target ratios
  - Verify sum of split sizes equals total samples
  - Test with 100 random datasets using Hypothesis
  - _Requirements: 4.1, 4.2_

---

### 4. Checkpoint - Verify Data Pipeline

- [ ] 4.1 Ensure all tests pass, ask the user if questions arise
  - Run dataset loading on actual `alphabets_dataset/`
  - Verify 37 classes and 5,000-5,200 samples loaded
  - Verify `dataset_statistics.txt` is created
  - Verify normalization produces expected output shapes
  - Ask user to confirm before proceeding to model implementation

---

### 5. Implement Neural Network Model

- [x] 5.1 Create `model.py` with AlphabetClassifier
  - Implement `AlphabetClassifier(nn.Module)` class
  - Define `__init__` with parameters: `input_dim=42`, `hidden_dim_1=128`, `hidden_dim_2=64`, `num_classes=37`, `dropout=0.3`
  - Create architecture: `fc1 (42→128)` → `BatchNorm1d` → `ReLU` → `Dropout` → `fc2 (128→64)` → `BatchNorm1d` → `ReLU` → `Dropout` → `fc3 (64→37)`
  - Implement `forward(x)` method returning logits (no softmax)
  - Add docstring explaining architecture and parameter count (~16K parameters)
  - Import all constants from `config.py`
  - _Requirements: 3.1, 3.2, 3.3_

- [ ]* 5.2 Write property test for softmax output constraint
  - **Property 5: Softmax Output Probability Constraint**
  - **Validates: Requirements 3.2**
  - Generate random 42-dimensional inputs
  - Pass through AlphabetClassifier to get logits
  - Apply softmax to logits
  - Verify probabilities sum to 1.0 within tolerance 1e-5
  - Verify each probability is in range [0, 1]
  - Test with 100 random inputs using Hypothesis
  - _Requirements: 3.2_

---

### 6. Implement Training Module

- [x] 6.1 Create data augmentation functions
  - Implement `augment_coords(coords: np.ndarray) -> np.ndarray` in `train.py`
  - Implement Gaussian noise: add `np.random.normal(0, AUGMENT_NOISE_STD, 42)`
  - Implement rotation: apply 2D rotation matrix to (x, y) pairs with angle ~ Uniform(-AUGMENT_ROTATION_DEG, +AUGMENT_ROTATION_DEG)
  - Implement scaling: multiply all coords by factor ~ Uniform(AUGMENT_SCALE_RANGE[0], AUGMENT_SCALE_RANGE[1])
  - Implement horizontal mirroring: negate all x-coordinates (even indices)
  - Randomly select 2-4 transforms per sample with probability 0.5 each
  - Return augmented (42,) array
  - _Requirements: 4.3_

- [x] 6.2 Implement training loop with early stopping
  - Implement `train(model, train_data, val_data, epochs, batch_size, lr, patience)` function
  - Create PyTorch DataLoader for train and validation sets
  - Initialize Adam optimizer with learning rate from config
  - Use CrossEntropyLoss as criterion
  - Implement training loop: for each epoch, iterate over batches, apply augmentation, compute loss, backpropagate
  - Implement validation loop: compute validation loss and accuracy without augmentation
  - Log epoch, train_loss, val_loss, val_accuracy to `training_log.csv`
  - Implement early stopping: track best validation loss, stop if no improvement for EARLY_STOPPING_PATIENCE epochs
  - Save best model weights to MODEL_PATH
  - Restore best weights after training completes
  - Print final validation and test accuracy
  - _Requirements: 3.4, 3.5, 3.6, 4.4, 4.5, 4.6_

- [ ]* 6.3 Write property test for augmentation dimensionality
  - **Property 7: Data Augmentation Preserves Dimensionality**
  - **Validates: Requirements 4.3**
  - Generate random normalized 42-dimensional coordinate vectors
  - Apply random combinations of augmentation transforms
  - Verify output shape is exactly (42,)
  - Verify augmentation parameters are within specified bounds
  - Test with 100 random inputs using Hypothesis
  - _Requirements: 4.3_

- [ ]* 6.4 Write property test for early stopping behavior
  - **Property 8: Early Stopping Triggers After Patience Epochs**
  - **Validates: Requirements 4.5**
  - Simulate training with controlled validation losses (increasing sequence)
  - Verify training stops at epoch N+P where N is last improvement and P is patience
  - Verify best model weights are restored
  - Test with various patience values and loss sequences
  - _Requirements: 4.5_

---

### 7. Checkpoint - Verify Training Pipeline

- [ ] 7.1 Ensure all tests pass, ask the user if questions arise
  - Train model for 5 epochs on small subset of data
  - Verify `alphabet_classifier.pt` is created
  - Verify `training_log.csv` contains expected columns
  - Verify model can be loaded and produces (batch, 37) output
  - Ask user to confirm before proceeding to evaluation

---

### 8. Implement Evaluation Module

- [x] 8.1 Create evaluation metrics computation
  - Implement `evaluate_model(model, test_data, label_map) -> dict` in `evaluate.py`
  - Load test set and run inference without augmentation
  - Compute overall accuracy using `(predictions == ground_truth).mean()`
  - Compute per-class precision, recall, F1-score using `sklearn.metrics.classification_report`
  - Generate 37×37 confusion matrix using `sklearn.metrics.confusion_matrix`
  - Identify top 5 most-confused alphabet pairs (highest off-diagonal values)
  - Return metrics dictionary with all computed values
  - _Requirements: 5.1, 5.2_

- [x] 8.2 Implement confusion matrix visualization
  - Implement `save_confusion_matrix(cm, labels, txt_path, png_path)` function
  - Save confusion matrix as text file with row/column labels
  - Generate heatmap using matplotlib + seaborn with colormap 'Blues'
  - Add annotations showing cell values
  - Rotate axis labels 90° for readability
  - Save as PNG image
  - _Requirements: 5.2, 5.3_

- [x] 8.3 Create evaluation report generation
  - Implement `save_evaluation_report(metrics, report_path)` function
  - Format report with overall accuracy, per-class metrics table, top 5 confused pairs
  - Add warning if test accuracy < 0.80
  - Save to `evaluation_report.txt`
  - _Requirements: 5.1, 5.4, 5.5_

- [ ]* 8.4 Write property test for confusion matrix correctness
  - **Property 9: Confusion Matrix Dimensions and Symmetry**
  - **Validates: Requirements 5.2**
  - Generate random predictions and ground truth labels from C classes
  - Compute confusion matrix
  - Verify shape is (C, C)
  - Verify sum of all elements equals number of test samples
  - Verify sum of row i equals number of samples with true label i
  - Test with 100 random prediction sets using Hypothesis
  - _Requirements: 5.2_

- [ ]* 8.5 Write property test for statistics computation
  - **Property 10: Dataset Statistics Computation Correctness**
  - **Validates: Requirements 8.2**
  - Generate random per-class sample counts
  - Compute min, max, mean, standard deviation
  - Verify min ≤ mean ≤ max
  - Verify std ≥ 0
  - Verify mean equals sum of counts divided by number of classes
  - Test with 100 random count distributions using Hypothesis
  - _Requirements: 8.2_

---

### 9. Implement Real-Time Demo Module

- [x] 9.1 Create MediaPipe hand detection integration
  - Implement `run_demo(model_path, camera_index, threshold)` in `demo.py`
  - Initialize MediaPipe Hands with `max_num_hands=1`, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`
  - Open webcam using `cv2.VideoCapture(camera_index)`
  - Check if webcam opened successfully, exit with error if not
  - Process each frame: convert BGR to RGB, run MediaPipe hand detection
  - Extract right hand landmarks by checking handedness label
  - Convert MediaPipe normalized coordinates to pixel coordinates: `(lm.x * width, lm.y * height)`
  - Flatten to (42,) array
  - _Requirements: 6.1, 6.2, 6.8, 10.1_

- [x] 9.2 Implement classification and confidence thresholding
  - Normalize extracted coordinates using `normalize_hand_coords()`
  - Run inference: convert to torch tensor, pass through model, apply softmax
  - Get maximum probability and predicted class index
  - If confidence >= threshold, map index to label using label_map
  - If confidence < threshold, set label to "Unknown"
  - Handle case when no hand detected: set label to "No hand detected"
  - Handle case when left hand detected: set label to "Please use right hand"
  - _Requirements: 6.3, 6.4, 6.5, 7.2_

- [x] 9.3 Create on-screen overlay visualization
  - Draw predicted label at position (10, 40) with font size 1.2
  - Color label green if confident (>= threshold), yellow if "Unknown"
  - Draw confidence score at position (10, 80)
  - Draw confidence bar from (10, 100) to (310, 120), width = confidence × 300
  - Color bar green if >= threshold, red otherwise
  - Draw vertical red line at threshold position on confidence bar
  - Draw "No hand detected" message in center if no hand found
  - Draw FPS counter in top-right corner
  - _Requirements: 6.3, 6.6, 7.3_

- [x] 9.4 Add main loop and exit handling
  - Display frame in window titled "PSL Alphabet Recognition"
  - Check for 'q' key press to exit
  - Release webcam and destroy all windows on exit
  - Compute and display FPS (target: 20 FPS)
  - _Requirements: 6.7, 6.9_

- [ ]* 9.5 Write property test for confidence threshold behavior
  - **Property 6: Confidence Threshold Determines Unknown Classification**
  - **Validates: Requirements 6.4, 7.2**
  - Generate random classifier outputs with varying max probabilities
  - Test with various threshold values
  - Verify "Unknown" returned when max_prob < threshold
  - Verify correct label returned when max_prob >= threshold
  - Test with 100 random outputs using Hypothesis
  - _Requirements: 6.4, 7.2_

---

### 10. Implement Coordinate Validation Module

- [x] 10.1 Create coordinate compatibility validator
  - Implement `validate_coordinate_compatibility(sample_json_path, tolerance)` in `validate_coordinates.py`
  - Load OpenPose JSON sample using `extract_right_hand_coords()`
  - Normalize OpenPose coordinates using `normalize_hand_coords()`
  - Simulate MediaPipe extraction (same raw coords, convert to pixel space)
  - Normalize MediaPipe coordinates
  - Compare normalized outputs element-wise
  - Compute maximum absolute difference
  - Return True if max_diff <= tolerance, False otherwise
  - Print mismatched indices if validation fails
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ]* 10.2 Write property test for coordinate validation round-trip
  - **Property 12: Coordinate Validation Round-Trip**
  - **Validates: Requirements 10.3**
  - Generate random raw hand coordinates
  - Extract and normalize via OpenPose path
  - Extract and normalize via simulated MediaPipe path
  - Verify normalized outputs match within tolerance 1e-3
  - Test with 100 random coordinate sets using Hypothesis
  - _Requirements: 10.3_

- [ ]* 10.3 Write property test for MediaPipe/OpenPose compatibility
  - **Property 3: MediaPipe and OpenPose Coordinate Compatibility**
  - **Validates: Requirements 2.4, 2.5, 10.1, 10.3**
  - Generate random raw hand coordinates
  - Normalize via OpenPose extraction path
  - Normalize via MediaPipe extraction path (with pixel conversion)
  - Verify outputs differ by at most 1e-3 in any dimension
  - Test with 100 random coordinate sets using Hypothesis
  - _Requirements: 2.4, 2.5, 10.1, 10.3_

---

### 11. Implement Unified Pipeline Orchestrator

- [x] 11.1 Create command-line argument parser
  - Implement `main()` function in `run_pipeline.py`
  - Add argument parser with `--mode` choices: `load`, `train`, `evaluate`, `demo`, `all`
  - Add `--threshold` argument (type float, default CONFIDENCE_THRESHOLD, range [0.0, 1.0])
  - Validate threshold is in valid range, exit with error if not
  - _Requirements: 9.1, 9.5, 7.4_

- [x] 11.2 Implement mode dispatch functions
  - Implement `run_load()`: load dataset, print statistics, save to file
  - Implement `run_train()`: load dataset, train model, save model and log
  - Implement `run_evaluate()`: load model, evaluate on test set, save reports
  - Implement `run_demo_mode(threshold)`: check model exists, run demo with specified threshold
  - Implement `run_all()`: execute load → train → evaluate → demo in sequence with timing
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 11.3 Add timing and progress reporting
  - Implement timing for each pipeline step in `run_all()` mode
  - Print start message, elapsed time, and completion message for each step
  - Print total pipeline time at end
  - Add model existence check for demo mode: exit with error if model not found
  - _Requirements: 9.3, 9.4_

- [ ]* 11.4 Write property test for threshold override
  - **Property 11: Command-Line Threshold Override**
  - **Validates: Requirements 7.4, 9.5**
  - Test with random threshold values in range [0.0, 1.0]
  - Verify threshold is applied correctly in demo mode
  - Verify predictions with confidence < T are classified as "Unknown"
  - Test with various threshold values
  - _Requirements: 7.4, 9.5_

---

### 12. Integration Testing and Final Validation

- [ ]* 12.1 Write integration test for full pipeline
  - Load actual `alphabets_dataset/` directory
  - Verify 37 classes and 5,000-5,200 samples loaded
  - Train model for 5 epochs on small subset
  - Verify `alphabet_classifier.pt` is created and loadable
  - Run inference on one sample, verify output shape is (37,)
  - Verify all output files are created: `training_log.csv`, `evaluation_report.txt`, `confusion_matrix.txt`, `confusion_matrix.png`, `dataset_statistics.txt`
  - _Requirements: 1.6, 3.6, 5.3, 5.4, 8.4_

- [ ]* 12.2 Write integration test for coordinate validation
  - Run `validate_coordinates.py` on actual OpenPose JSON sample
  - Verify validation passes (max_diff <= 1e-3)
  - Test with multiple samples from different alphabet classes
  - _Requirements: 10.3, 10.4_

---

### 13. Final Checkpoint - Complete System Validation

- [x] 13.1 Ensure all tests pass, ask the user if questions arise
  - Run full pipeline with `python run_pipeline.py --mode all`
  - Verify all output files are created
  - Verify model achieves reasonable validation accuracy (> 70%)
  - Verify demo opens webcam and displays predictions
  - Verify coordinate validation passes
  - Ask user to confirm system is ready for deployment

---

## Notes

- **Tasks marked with `*` are optional** and can be skipped for faster MVP
- **Each task references specific requirements** for traceability (e.g., _Requirements: 1.2, 1.3_)
- **Checkpoints ensure incremental validation** at major milestones
- **Property tests validate universal correctness properties** using Hypothesis (100 iterations each)
- **Unit tests validate specific examples and edge cases**
- **Implementation order follows dependency chain**: config → preprocessor → dataset_loader → model → train → evaluate/demo → orchestrator
- **All modules import constants from `config.py`** (single source of truth)
- **Error handling is included in each module** for robustness
- **The system uses Python 3.8+** with PyTorch, NumPy, OpenCV, MediaPipe, scikit-learn, matplotlib, seaborn

## Property-Based Test Summary

| Property | Description | Validates Requirements |
|----------|-------------|----------------------|
| Property 1 | Coordinate extraction pipeline correctness | 1.2, 1.3, 1.4 |
| Property 2 | Normalization translation and scale invariance | 2.1, 2.2 |
| Property 3 | MediaPipe and OpenPose coordinate compatibility | 2.4, 2.5, 10.1, 10.3 |
| Property 4 | Stratified dataset splitting preserves class distribution | 4.1, 4.2 |
| Property 5 | Softmax output probability constraint | 3.2 |
| Property 6 | Confidence threshold determines unknown classification | 6.4, 7.2 |
| Property 7 | Data augmentation preserves dimensionality | 4.3 |
| Property 8 | Early stopping triggers after patience epochs | 4.5 |
| Property 9 | Confusion matrix dimensions and symmetry | 5.2 |
| Property 10 | Dataset statistics computation correctness | 8.2 |
| Property 11 | Command-line threshold override | 7.4, 9.5 |
| Property 12 | Coordinate validation round-trip | 10.3 |

## Expected Outputs

After completing all tasks, the following files will be generated:

- `config.py` — Configuration constants
- `preprocessor.py` — Normalization functions
- `dataset_loader.py` — Dataset loading and splitting
- `model.py` — AlphabetClassifier neural network
- `train.py` — Training loop with augmentation
- `evaluate.py` — Evaluation metrics and confusion matrix
- `demo.py` — Real-time webcam demo
- `validate_coordinates.py` — Coordinate compatibility validator
- `run_pipeline.py` — Unified entry point
- `alphabet_classifier.pt` — Trained model weights
- `training_log.csv` — Training history
- `evaluation_report.txt` — Test set metrics
- `confusion_matrix.txt` — Confusion matrix (text)
- `confusion_matrix.png` — Confusion matrix (heatmap)
- `dataset_statistics.txt` — Dataset statistics

## Execution Instructions

To begin implementation, open `tasks.md` and click "Start task" next to task items. The system will guide you through each step with incremental progress validation.

**This workflow creates design and planning artifacts only.** Implementation begins after task creation is complete.
