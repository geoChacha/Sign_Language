# Implementation Plan: ANN-Based PSL Alphabet Recognition System

## Overview

This implementation plan breaks down the development of an Artificial Neural Network (ANN) based Pakistan Sign Language (PSL) alphabet recognition system into discrete coding tasks. The system uses a feedforward neural network to classify static hand gestures from 2D hand keypoint coordinates, with training on OpenPose data and real-time inference using MediaPipe.

The implementation follows a bottom-up approach: starting with data loading and preprocessing, building the neural network architecture, implementing training and evaluation pipelines, and finally creating the real-time inference demo.

## Tasks

- [x] 1. Set up project structure and configuration
  - Create `alphabet_recognition/` directory structure
  - Create `config.py` with all hyperparameters and paths
  - Define constants: `DATASET_ROOT`, `MODEL_PATH`, `INPUT_DIM=42`, `HIDDEN_DIM_1=128`, `HIDDEN_DIM_2=64`, `DROPOUT=0.3`, `LEARNING_RATE=0.001`, `BATCH_SIZE=32`, `MAX_EPOCHS=50`, `PATIENCE=10`, `CONFIDENCE_THRESHOLD=0.70`
  - Define augmentation parameters: `NOISE_STD=0.02`, `ROTATION_RANGE=15`, `SCALE_RANGE=(0.9, 1.1)`
  - Define split ratios: `TRAIN_SPLIT=0.70`, `VAL_SPLIT=0.15`, `TEST_SPLIT=0.15`
  - Add documentation comments for each parameter
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 2. Implement coordinate normalization
  - [x] 2.1 Create `preprocessor.py` module
    - Implement `normalize_hand_coords(coords: np.ndarray) -> np.ndarray` function
    - Handle both (42,) and (21, 2) input shapes
    - Compute centroid of all 21 landmarks
    - Translate coordinates to center at origin
    - Compute bounding box (max - min for x and y)
    - Scale by maximum of bounding box width and height
    - Return flattened (42,) array with dtype float32
    - Handle degenerate case (all landmarks identical) by returning zero vector
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  
  - [ ]* 2.2 Write property test for translation invariance
    - **Property 1: Translation Invariance of Normalization**
    - **Validates: Requirements 3.1, 3.2, 3.3**
    - Use hypothesis to generate random coordinates and translation offsets
    - Verify `normalize(coords + offset) ≈ normalize(coords)` within tolerance 1e-5
    - Run 100+ iterations
  
  - [ ]* 2.3 Write property test for scale invariance
    - **Property 2: Scale Invariance of Normalization**
    - **Validates: Requirements 3.3**
    - Use hypothesis to generate random coordinates and positive scale factors
    - Verify `normalize(coords * scale) ≈ normalize(coords)` within tolerance 1e-5
    - Run 100+ iterations
  
  - [ ]* 2.4 Write property test for output shape consistency
    - **Property 3: Normalization Output Shape Consistency**
    - **Validates: Requirements 3.6**
    - Test both (42,) and (21, 2) input shapes
    - Verify output is always (42,) with dtype float32
    - Run 100+ iterations
  
  - [ ]* 2.5 Write property test for coordinate range validation
    - **Property 13: Coordinate Validation Range Checking**
    - **Validates: Requirements 15.2, 15.3**
    - Generate random raw coordinates, normalize them
    - Verify all values in [-2, 2] and at least 90% in [-1, 1]
    - Run 100+ iterations
  
  - [ ]* 2.6 Write unit tests for edge cases
    - Test degenerate hand pose (all landmarks identical)
    - Test invalid input shapes
    - Test zero coordinates

- [x] 3. Implement dataset loading and parsing
  - [x] 3.1 Create `dataset_loader.py` module
    - Implement `extract_right_hand_coords(json_path: Path) -> Optional[np.ndarray]` function
    - Parse OpenPose JSON files
    - Extract `hand_right_keypoints_2d` from first person in `people` array
    - Convert 63-element array to 42-element array (extract x, y pairs, discard confidence)
    - Return None if people array is empty
    - Return None if all coordinates are zero
    - Log warnings for invalid files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 11.4_
  
  - [x] 3.2 Implement dataset loading and label encoding
    - Implement `load_dataset(root_dir, train_split, val_split, test_split)` function
    - Recursively scan `PSL_dataset/datasets/alphabets_dataset/` directory
    - Load all JSON files using `extract_right_hand_coords`
    - Create integer labels from folder names in alphabetical order
    - Create label mapping dictionary (int → alphabet string)
    - Support dynamic class count detection
    - Log total samples and classes loaded
    - _Requirements: 1.6, 1.7, 1.8, 1.9, 1.10_
  
  - [x] 3.3 Implement stratified dataset splitting
    - Use scikit-learn's `train_test_split` with stratification
    - Split into train (70%), validation (15%), test (15%)
    - Use fixed random seed (42) for reproducibility
    - Return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_map
    - Log warning if any class has fewer than 50 samples
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x] 3.4 Implement dataset statistics reporting
    - Compute total samples, total classes, per-class counts
    - Compute min, max, mean, std of samples per class
    - Save statistics to `dataset_statistics.txt` with UTF-8 encoding
    - Print statistics to console
    - Include warnings for classes with <100 samples
    - Note that current dataset is incomplete (23 classes, designed for 38-40)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_
  
  - [ ]* 3.5 Write property test for coordinate array conversion
    - **Property 4: Coordinate Array Conversion Correctness**
    - **Validates: Requirements 1.3**
    - Generate random 63-element arrays
    - Verify output has 42 elements
    - Verify `output[2*i] == input[3*i]` and `output[2*i+1] == input[3*i+1]` for all i in [0, 20]
    - Run 100+ iterations
  
  - [ ]* 3.6 Write property test for label encoding ordering
    - **Property 5: Label Encoding Alphabetical Ordering**
    - **Validates: Requirements 1.6**
    - Generate random sets of folder names
    - Create label encoding
    - Verify labels are assigned in alphabetical order
    - Run 100+ iterations
  
  - [ ]* 3.7 Write property test for label map inversion
    - **Property 6: Label Map Inversion Correctness**
    - **Validates: Requirements 1.7**
    - Generate random label encodings
    - Create label map (inverse)
    - Verify `label_map[label_encoding[name]] == name` for all names
    - Run 100+ iterations
  
  - [ ]* 3.8 Write property test for split proportions
    - **Property 7: Dataset Split Proportion Correctness**
    - **Validates: Requirements 2.1**
    - Generate random datasets (100-10000 samples)
    - Perform splitting
    - Verify actual proportions within 5% of target
    - Run 100+ iterations
  
  - [ ]* 3.9 Write property test for stratified sampling
    - **Property 8: Stratified Split Class Distribution Preservation**
    - **Validates: Requirements 2.2**
    - Generate random datasets with known class distributions
    - Perform stratified splitting
    - Verify class proportions in each split within 10% of original
    - Run 100+ iterations
  
  - [ ]* 3.10 Write unit tests for dataset loading
    - Test empty people array returns None
    - Test all-zero coordinates returns None
    - Test JSON parsing errors are handled
    - Test warning for classes with <50 samples

- [x] 4. Checkpoint - Verify data loading and preprocessing
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement data augmentation
  - [x] 5.1 Create augmentation functions in `preprocessor.py`
    - Implement `add_gaussian_noise(coords, std=0.02)` function
    - Implement `rotate_coordinates(coords, angle_degrees)` function (rotate around wrist)
    - Implement `scale_coordinates(coords, scale_factor)` function
    - Implement `augment_coordinates(coords)` function that applies random transformations
    - Apply transformations in random order
    - Use random rotation in [-15°, +15°]
    - Use random scaling in [0.9, 1.1]
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 4.6_
  
  - [ ]* 5.2 Write property test for augmentation parameter bounds
    - **Property 9: Data Augmentation Parameter Bounds**
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - Apply augmentation 100+ times
    - Track noise std, rotation angles, scale factors
    - Verify all within specified bounds
    - Run 100+ iterations
  
  - [ ]* 5.3 Write property test for rotation preserves wrist
    - **Property 10: Rotation Preserves Wrist Position**
    - **Validates: Requirements 4.6**
    - Generate normalized coordinates (wrist at origin)
    - Apply rotation augmentation
    - Verify wrist remains within 0.01 of origin
    - Run 100+ iterations
  
  - [ ]* 5.4 Write unit tests for augmentation
    - Test noise standard deviation is approximately 0.02
    - Test rotation angles are in range
    - Test scale factors are in range

- [x] 6. Implement neural network architecture
  - [x] 6.1 Create `model.py` module
    - Implement `AlphabetClassifier(nn.Module)` class
    - Constructor parameters: `input_dim=42`, `hidden_dim_1=128`, `hidden_dim_2=64`, `num_classes`, `dropout=0.3`
    - Layer 1: Linear(42, 128) + BatchNorm1d(128) + ReLU + Dropout(0.3)
    - Layer 2: Linear(128, 64) + BatchNorm1d(64) + ReLU + Dropout(0.3)
    - Layer 3: Linear(64, num_classes)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_
  
  - [x] 6.2 Implement forward pass and prediction methods
    - Implement `forward(x)` method returning raw logits
    - Implement `predict_proba(x)` method returning softmax probabilities
    - Implement `predict(x)` method returning class indices (argmax)
    - _Requirements: 5.9_
  
  - [ ]* 6.3 Write property test for batch inference shape consistency
    - **Property 12: Batch Inference Output Shape Consistency**
    - **Validates: Requirements 18.1, 18.2**
    - Generate random batches of varying sizes (1, 16, 32, 64, 128)
    - Run forward pass
    - Verify output shape is (batch_size, num_classes)
    - Run 100+ iterations
  
  - [ ]* 6.4 Write unit tests for model architecture
    - Test model has correct number of parameters (~15,000 for 23 classes)
    - Test input dimension is 42
    - Test hidden layer dimensions are 128 and 64
    - Test output dimension matches num_classes
    - Test forward pass with single sample
    - Test forward pass with batch

- [x] 7. Implement training pipeline
  - [x] 7.1 Create `train.py` script
    - Load dataset using `load_dataset()`
    - Apply normalization to all samples
    - Create PyTorch DataLoaders for train/val sets with batch_size=32
    - Instantiate AlphabetClassifier with dynamic num_classes from dataset
    - Use Adam optimizer with learning rate 0.001
    - Use CrossEntropyLoss
    - _Requirements: 6.1, 6.2, 6.3, 19.1, 19.2, 19.3_
  
  - [x] 7.2 Implement training loop with early stopping
    - Train for maximum 50 epochs
    - Apply augmentation to training batches only
    - Compute training loss per epoch
    - Compute validation loss and accuracy per epoch
    - Implement early stopping with patience=10 based on validation loss
    - Track best model weights
    - Restore best weights when early stopping triggers
    - _Requirements: 6.4, 6.5, 6.6, 4.4_
  
  - [x] 7.3 Implement logging and model saving
    - Log training loss, validation loss, validation accuracy after each epoch
    - Save metrics to `training_log.csv` with header
    - Format metrics to 4 decimal places
    - Save best model state dict to `alphabet_classifier.pt`
    - Log informational messages for major steps
    - Log when early stopping triggers with epoch number
    - _Requirements: 6.7, 6.8, 6.9, 11.1, 11.5, 11.6, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  
  - [ ]* 7.4 Write integration test for training pipeline
    - Load actual dataset (23 classes)
    - Train model for 5 epochs (smoke test)
    - Verify model file is saved
    - Verify training log CSV is created
    - Verify training and validation losses decrease

- [x] 8. Checkpoint - Verify training pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement model evaluation
  - [x] 9.1 Create `evaluate.py` script
    - Load trained model from `alphabet_classifier.pt`
    - Load test set from dataset
    - Create DataLoader for test set with batch_size=32
    - Run inference on all test samples in batches
    - _Requirements: 12.3, 12.4, 18.3, 18.4, 18.5_
  
  - [x] 9.2 Compute evaluation metrics
    - Compute overall accuracy
    - Compute per-class precision, recall, F1-score using scikit-learn
    - Compute macro-averaged and weighted-averaged F1-scores
    - Generate confusion matrix
    - Identify classes with F1-score < 0.70 as problematic
    - _Requirements: 7.1, 7.2, 7.3, 7.6, 7.7_
  
  - [x] 9.3 Save evaluation outputs
    - Save detailed evaluation report to `evaluation_report.txt`
    - Include overall accuracy, per-class metrics, macro/weighted F1
    - List problematic classes
    - Save confusion matrix as text to `confusion_matrix.txt`
    - Generate confusion matrix heatmap with matplotlib
    - Save heatmap to `confusion_matrix.png`
    - Use alphabet labels (not integer indices) on axes
    - _Requirements: 7.4, 7.5, 7.8_
  
  - [ ]* 9.4 Write integration test for evaluation pipeline
    - Load trained model
    - Run evaluation on test set
    - Verify evaluation report is generated
    - Verify confusion matrix files are created
    - Check accuracy is reasonable (>0.50 for 23 classes)

- [x] 10. Implement model persistence and loading
  - [x] 10.1 Add model saving functionality
    - Save model state dict using `torch.save()`
    - Save in PyTorch .pt format
    - Save to `alphabet_classifier.pt`
    - _Requirements: 12.1, 12.2_
  
  - [x] 10.2 Add model loading functionality
    - Implement model loading in `evaluate.py` and `demo.py`
    - Instantiate AlphabetClassifier with correct architecture
    - Load state dict using `torch.load()`
    - Set model to eval mode
    - Handle FileNotFoundError with descriptive message
    - Handle architecture mismatch with RuntimeError
    - Support CPU loading even if trained on GPU
    - _Requirements: 12.3, 12.4, 12.5, 12.6, 12.7_
  
  - [ ]* 10.3 Write property test for model persistence
    - **Property 11: Model State Dictionary Round-Trip Preservation**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
    - Create models with random weights
    - Save to temporary file
    - Load back
    - Verify all parameters match within 1e-6
    - Run 100+ iterations
  
  - [ ]* 10.4 Write integration test for model persistence
    - Train small model
    - Save to file
    - Load from file
    - Verify predictions match
  
  - [ ]* 10.5 Write unit tests for error handling
    - Test FileNotFoundError when model file missing
    - Test RuntimeError on architecture mismatch

- [ ] 11. Implement real-time inference with MediaPipe
  - [x] 11.1 Create `demo.py` script
    - Import MediaPipe Hands
    - Load trained model from `alphabet_classifier.pt`
    - Load label map (need to save/load with model)
    - Initialize MediaPipe Hands detector
    - Open webcam using OpenCV
    - Handle webcam access failure with descriptive error
    - _Requirements: 8.1, 12.3_
  
  - [x] 11.2 Implement frame processing loop
    - Capture frames from webcam in loop
    - Process each frame with MediaPipe to detect hand landmarks
    - Extract 21 landmarks with x, y coordinates (42 values)
    - Normalize coordinates using `normalize_hand_coords()`
    - Convert to PyTorch tensor
    - Run model forward pass to get probabilities
    - Get predicted class and confidence
    - _Requirements: 8.2, 8.3, 8.4, 8.9_
  
  - [x] 11.3 Implement prediction display and visualization
    - Display predicted alphabet if confidence ≥ 0.70
    - Display "Low Confidence" if confidence < 0.70
    - Display "No hand detected" if no hand found
    - Show confidence percentage
    - Use green text for confidence ≥ 0.85
    - Use yellow text for confidence 0.70-0.84
    - Render hand landmarks as circles on frame
    - Draw connections between landmarks
    - Display instructions "Press 'q' to quit"
    - Quit when 'q' key is pressed
    - _Requirements: 8.5, 8.6, 8.7, 8.8, 8.10, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [x] 12. Implement coordinate validation utility
  - [x] 12.1 Create `validate_coordinates.py` script
    - Load random sample of 100 coordinates from training set
    - Verify all values are within [-2, 2]
    - Verify wrist landmark is near origin (within 0.1)
    - Compute and display bounding box dimensions
    - Print warnings for coordinates outside expected range
    - Print summary of validation results
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_

- [x] 13. Create command-line interface scripts
  - [x] 13.1 Finalize `train.py` as standalone script
    - Add `if __name__ == "__main__":` block
    - Load configuration from `config.py`
    - Call training pipeline
    - Print completion message
    - _Requirements: 20.1, 20.4_
  
  - [x] 13.2 Finalize `evaluate.py` as standalone script
    - Add `if __name__ == "__main__":` block
    - Check if model file exists, print error if not
    - Load model automatically
    - Call evaluation pipeline
    - Print completion message
    - _Requirements: 20.2, 20.5, 20.7_
  
  - [x] 13.3 Finalize `demo.py` as standalone script
    - Add `if __name__ == "__main__":` block
    - Check if model file exists, print error if not
    - Load model automatically
    - Launch webcam demo
    - _Requirements: 20.3, 20.6, 20.7_
  
  - [x] 13.4 Create `run_pipeline.py` orchestration script
    - Execute `train.py`
    - Execute `evaluate.py`
    - Optionally execute `demo.py` (with user confirmation)
    - Handle errors gracefully
    - _Requirements: 20.8_

- [ ] 14. Create requirements.txt and documentation
  - [ ] 14.1 Create `requirements.txt` file
    - Add torch and torchvision
    - Add mediapipe
    - Add opencv-python
    - Add numpy
    - Add scikit-learn
    - Add matplotlib
    - Add hypothesis (for property-based testing)
    - Add pytest (for unit testing)
    - Specify Python 3.8+ compatibility
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8_
  
  - [x] 14.2 Create README.md with usage instructions
    - Document system overview
    - Document installation steps
    - Document how to run training, evaluation, and demo
    - Document configuration options
    - Document dataset structure requirements
    - Note current dataset has 23 classes, designed for 38-40

- [x] 15. Implement dataset extensibility features
  - [x] 15.1 Add class count detection and model compatibility
    - Automatically detect number of classes from dataset directory
    - Save num_classes alongside model weights
    - When loading model, verify class count matches dataset
    - Print warning if mismatch detected
    - Save label_map alongside model for consistency
    - Log number of detected classes during initialization
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9_
  
  - [ ]* 15.2 Write integration test for dataset expansion
    - Test system with different numbers of classes
    - Verify model adapts automatically
    - Verify label map consistency

- [ ] 16. Final checkpoint - End-to-end testing
  - Run complete pipeline: train → evaluate → demo
  - Verify all output files are generated
  - Verify model achieves reasonable accuracy (>0.60 for 23 classes)
  - Verify webcam demo runs smoothly
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties across many random inputs
- Unit tests validate specific examples and edge cases
- Integration tests verify end-to-end workflows with real data
- The system is designed to support 38-40 Urdu alphabet classes but currently works with 23 classes
- All code should include appropriate error handling and logging as specified in Requirements 11
- Use fixed random seed (42) throughout for reproducibility (Requirement 19)
