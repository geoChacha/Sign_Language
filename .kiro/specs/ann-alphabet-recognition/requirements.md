# Requirements Document

## Introduction

This document specifies the requirements for an Artificial Neural Network (ANN) based Pakistan Sign Language (PSL) alphabet recognition system. The system will classify static hand poses representing the full Urdu alphabet (38-40 classes) using hand keypoint coordinates extracted from images. The system focuses exclusively on alphabet recognition (no word or sentence recognition) and uses a feedforward neural network architecture trained on OpenPose hand keypoints with real-time inference capabilities via MediaPipe.

**Note:** The current dataset contains only 23 alphabet classes with approximately 1,720 samples. The system is designed to be extensible to support all 38-40 Urdu letters when additional data is collected.

## Glossary

- **ANN_Classifier**: The feedforward artificial neural network that classifies hand poses into alphabet classes
- **Dataset_Loader**: Component that loads and parses OpenPose JSON files from the alphabets dataset
- **Coordinate_Normalizer**: Component that normalizes hand coordinates for translation and scale invariance
- **Training_Pipeline**: Component that trains the ANN_Classifier using the training dataset
- **Augmentation_Engine**: Component that applies data augmentation transformations during training
- **Evaluation_Module**: Component that computes performance metrics and confusion matrices
- **Inference_Engine**: Component that performs real-time classification on webcam input
- **MediaPipe_Detector**: Real-time hand landmark detection system used during inference
- **OpenPose_Format**: JSON format containing hand_right_keypoints_2d array with 21 landmarks × 3 values (x, y, confidence)
- **Hand_Keypoints**: 21 anatomical landmarks on the hand (wrist, thumb joints, finger joints, fingertips)
- **Normalized_Coordinates**: Hand coordinates transformed to be translation and scale invariant
- **Confusion_Matrix**: Matrix showing predicted vs actual classifications for error analysis
- **Confidence_Threshold**: Minimum probability required for a prediction to be considered confident

## Requirements

### Requirement 1: Dataset Loading and Parsing

**User Story:** As a machine learning engineer, I want to load hand keypoint data from OpenPose JSON files, so that I can train the ANN classifier on the PSL alphabet dataset.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL load JSON files from the PSL_dataset/datasets/alphabets_dataset directory structure
2. WHEN a JSON file is parsed, THE Dataset_Loader SHALL extract the hand_right_keypoints_2d array from the first person in the people array
3. WHEN the hand_right_keypoints_2d array is extracted, THE Dataset_Loader SHALL convert the 63-element array (21 landmarks × 3 values) into 42 coordinate values by extracting x and y pairs and discarding confidence values
4. IF a JSON file has an empty people array, THEN THE Dataset_Loader SHALL skip that file and log a warning
5. IF a JSON file has all-zero coordinates, THEN THE Dataset_Loader SHALL skip that file and log a warning
6. THE Dataset_Loader SHALL create integer labels from alphabet folder names in alphabetical order
7. THE Dataset_Loader SHALL return a label mapping dictionary that maps integer labels to alphabet strings
8. THE Dataset_Loader SHALL load all available alphabet classes from the dataset (currently 23 classes, designed to support 38-40 classes)
9. THE Dataset_Loader SHALL load between 1,500 and 2,000 total samples from the current dataset
10. THE Dataset_Loader SHALL support dynamic class count detection to accommodate dataset expansion

### Requirement 2: Dataset Splitting

**User Story:** As a machine learning engineer, I want stratified train/validation/test splits, so that each split has proportional representation of all alphabet classes.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL split the dataset into training (70%), validation (15%), and test (15%) sets
2. WHEN creating splits, THE Dataset_Loader SHALL use stratified sampling to maintain class distribution across all splits
3. THE Dataset_Loader SHALL use a fixed random seed (42) for reproducible splits
4. THE Dataset_Loader SHALL return training, validation, and test sets as separate (X, y) tuples
5. WHEN a class has fewer than 50 samples, THE Dataset_Loader SHALL log a warning about potential training instability

### Requirement 3: Coordinate Normalization

**User Story:** As a machine learning engineer, I want hand coordinates normalized for translation and scale invariance, so that the model generalizes across different hand positions and sizes.

#### Acceptance Criteria

1. WHEN raw coordinates are provided, THE Coordinate_Normalizer SHALL translate coordinates so the wrist (landmark 0) is at the origin (0, 0)
2. WHEN coordinates are translated, THE Coordinate_Normalizer SHALL compute the bounding box of all landmarks
3. WHEN the bounding box is computed, THE Coordinate_Normalizer SHALL scale coordinates by dividing by the maximum of bounding box width and height
4. THE Coordinate_Normalizer SHALL ensure normalized coordinates fit within the range [-1, 1]
5. THE Coordinate_Normalizer SHALL preserve the aspect ratio of the hand during normalization
6. THE Coordinate_Normalizer SHALL return a 42-element array of normalized coordinates

### Requirement 4: Data Augmentation

**User Story:** As a machine learning engineer, I want data augmentation during training, so that the model is robust to natural variations in hand pose.

#### Acceptance Criteria

1. WHEN training data is augmented, THE Augmentation_Engine SHALL add Gaussian noise with standard deviation 0.02 to coordinates
2. WHEN training data is augmented, THE Augmentation_Engine SHALL apply random rotation between -15 and +15 degrees around the wrist point
3. WHEN training data is augmented, THE Augmentation_Engine SHALL apply random uniform scaling between 0.9 and 1.1
4. THE Augmentation_Engine SHALL apply augmentation only to training data, not to validation or test data
5. THE Augmentation_Engine SHALL apply augmentation transformations in random order for each sample
6. WHEN rotation is applied, THE Augmentation_Engine SHALL rotate all landmarks around the wrist (landmark 0) as the center of rotation

### Requirement 5: Neural Network Architecture

**User Story:** As a machine learning engineer, I want a feedforward neural network architecture optimized for hand pose classification, so that the system achieves high accuracy on alphabet recognition.

#### Acceptance Criteria

1. THE ANN_Classifier SHALL have an input layer accepting 42-dimensional coordinate vectors
2. THE ANN_Classifier SHALL have a first hidden layer with 128 neurons
3. THE ANN_Classifier SHALL have a second hidden layer with 64 neurons
4. THE ANN_Classifier SHALL have an output layer with a configurable number of neurons matching the number of alphabet classes (38-40 for full Urdu alphabet, currently 23 for available dataset)
5. THE ANN_Classifier SHALL determine the output layer size dynamically based on the number of classes in the loaded dataset
6. THE ANN_Classifier SHALL apply batch normalization after each linear layer
7. THE ANN_Classifier SHALL use ReLU activation functions after batch normalization
8. THE ANN_Classifier SHALL apply dropout with probability 0.3 after the first and second hidden layers
9. THE ANN_Classifier SHALL output raw logits (no softmax in forward pass)
10. THE ANN_Classifier SHALL have between 10,000 and 25,000 trainable parameters depending on the number of output classes

### Requirement 6: Training Process

**User Story:** As a machine learning engineer, I want a robust training pipeline with early stopping, so that the model converges efficiently without overfitting.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL use the Adam optimizer with learning rate 0.001
2. THE Training_Pipeline SHALL use cross-entropy loss as the training objective
3. THE Training_Pipeline SHALL train in mini-batches of size 32
4. THE Training_Pipeline SHALL train for a maximum of 50 epochs
5. THE Training_Pipeline SHALL implement early stopping with patience of 10 epochs based on validation loss
6. WHEN validation loss does not improve for 10 consecutive epochs, THE Training_Pipeline SHALL stop training and restore the best model weights
7. THE Training_Pipeline SHALL log training loss, validation loss, and validation accuracy after each epoch
8. THE Training_Pipeline SHALL save the trained model weights to a file named alphabet_classifier.pt
9. THE Training_Pipeline SHALL save training history to a CSV file named training_log.csv

### Requirement 7: Model Evaluation

**User Story:** As a machine learning engineer, I want comprehensive evaluation metrics, so that I can assess model performance and identify problematic alphabet classes.

#### Acceptance Criteria

1. THE Evaluation_Module SHALL compute overall accuracy on the test set
2. THE Evaluation_Module SHALL compute per-class precision, recall, and F1-score
3. THE Evaluation_Module SHALL generate a confusion matrix showing predicted vs actual classes
4. THE Evaluation_Module SHALL save the confusion matrix as both a text file and a PNG visualization
5. THE Evaluation_Module SHALL save a detailed evaluation report including all metrics to evaluation_report.txt
6. WHEN per-class metrics are computed, THE Evaluation_Module SHALL identify classes with F1-score below 0.70 as problematic
7. THE Evaluation_Module SHALL compute macro-averaged and weighted-averaged F1-scores
8. THE Evaluation_Module SHALL display the confusion matrix with alphabet labels (not integer indices)

### Requirement 8: Real-Time Inference with MediaPipe

**User Story:** As a user, I want real-time alphabet recognition from my webcam, so that I can test the system with live hand poses.

#### Acceptance Criteria

1. THE Inference_Engine SHALL use MediaPipe Hands to detect hand landmarks in real-time from webcam input
2. WHEN a hand is detected, THE Inference_Engine SHALL extract 21 landmarks with x and y coordinates
3. WHEN landmarks are extracted, THE Inference_Engine SHALL normalize coordinates using the same normalization as training
4. WHEN normalized coordinates are obtained, THE Inference_Engine SHALL pass them through the ANN_Classifier to get class probabilities
5. WHEN class probabilities are computed, THE Inference_Engine SHALL display the predicted alphabet if confidence exceeds 0.70
6. WHEN confidence is below 0.70, THE Inference_Engine SHALL display "Low Confidence" instead of a prediction
7. THE Inference_Engine SHALL display the confidence percentage alongside the predicted alphabet
8. THE Inference_Engine SHALL render hand landmarks on the video frame for visual feedback
9. THE Inference_Engine SHALL process video frames at a minimum of 15 frames per second
10. WHEN the user presses the 'q' key, THE Inference_Engine SHALL terminate the webcam demo

### Requirement 9: Webcam Demo Interface

**User Story:** As a user, I want a visual interface showing predictions and confidence, so that I can understand what the system is recognizing.

#### Acceptance Criteria

1. THE Inference_Engine SHALL display the webcam video feed in a window titled "PSL Alphabet Recognition"
2. THE Inference_Engine SHALL overlay the predicted alphabet in large text on the video frame
3. THE Inference_Engine SHALL overlay the confidence percentage below the predicted alphabet
4. THE Inference_Engine SHALL draw hand landmarks as circles on the detected hand
5. THE Inference_Engine SHALL draw connections between hand landmarks to visualize hand skeleton
6. WHEN no hand is detected, THE Inference_Engine SHALL display "No hand detected" on the video frame
7. THE Inference_Engine SHALL display instructions "Press 'q' to quit" on the video frame
8. THE Inference_Engine SHALL use green color for high-confidence predictions (≥0.85) and yellow color for medium-confidence predictions (0.70-0.84)

### Requirement 10: System Configuration

**User Story:** As a developer, I want all hyperparameters and paths centralized in a configuration file, so that I can easily modify system settings without changing code.

#### Acceptance Criteria

1. THE system SHALL define all hyperparameters (learning rate, batch size, epochs, dropout, etc.) in a config.py file
2. THE system SHALL define all file paths (dataset root, model path, output paths) in the config.py file
3. THE system SHALL define the neural network architecture dimensions (input size: 42, hidden layers: 128 and 64, output size: configurable based on dataset) in the config.py file
4. THE system SHALL define data augmentation parameters (noise std, rotation range, scale range) in the config.py file
5. THE system SHALL define the confidence threshold for inference in the config.py file
6. THE system SHALL import configuration values from config.py in all other modules
7. THE config.py file SHALL include documentation comments explaining each configuration parameter
8. THE config.py file SHALL note that the output layer size is determined dynamically from the dataset

### Requirement 11: Error Handling and Logging

**User Story:** As a developer, I want comprehensive error handling and logging, so that I can diagnose issues during training and inference.

#### Acceptance Criteria

1. THE system SHALL log informational messages for major pipeline steps (loading data, starting training, saving model)
2. THE system SHALL log warnings for data quality issues (missing files, invalid JSON, zero coordinates)
3. THE system SHALL log errors for critical failures (file not found, model loading failure, GPU errors)
4. WHEN a JSON file fails to parse, THE Dataset_Loader SHALL log the file path and error message
5. WHEN training completes, THE Training_Pipeline SHALL log the final training and validation metrics
6. WHEN early stopping triggers, THE Training_Pipeline SHALL log the epoch number and reason
7. THE system SHALL use Python's logging module with appropriate log levels (INFO, WARNING, ERROR)
8. THE system SHALL print log messages to the console during execution

### Requirement 12: Model Persistence and Loading

**User Story:** As a developer, I want to save and load trained models, so that I can reuse trained models for inference without retraining.

#### Acceptance Criteria

1. WHEN training completes, THE Training_Pipeline SHALL save the model state dictionary to alphabet_classifier.pt
2. THE Training_Pipeline SHALL save the model in PyTorch's standard .pt format
3. WHEN loading a model for inference, THE Inference_Engine SHALL load the state dictionary from alphabet_classifier.pt
4. WHEN loading a model, THE Inference_Engine SHALL instantiate the ANN_Classifier with the same architecture as training
5. IF the model file does not exist, THE Inference_Engine SHALL raise a FileNotFoundError with a descriptive message
6. IF the model architecture does not match the saved weights, THE system SHALL raise a RuntimeError with a descriptive message
7. THE system SHALL support loading models on CPU even if trained on GPU

### Requirement 13: Dependencies and Environment

**User Story:** As a developer, I want a requirements.txt file specifying all dependencies, so that I can set up the environment consistently.

#### Acceptance Criteria

1. THE system SHALL provide a requirements.txt file listing all Python package dependencies
2. THE requirements.txt file SHALL specify PyTorch (torch and torchvision)
3. THE requirements.txt file SHALL specify MediaPipe for hand detection
4. THE requirements.txt file SHALL specify OpenCV (opencv-python) for video processing
5. THE requirements.txt file SHALL specify NumPy for numerical operations
6. THE requirements.txt file SHALL specify scikit-learn for dataset splitting and metrics
7. THE requirements.txt file SHALL specify Matplotlib for confusion matrix visualization
8. THE system SHALL support Python 3.8 or higher
9. THE system SHALL support Windows operating system

### Requirement 14: Dataset Statistics Reporting

**User Story:** As a machine learning engineer, I want detailed dataset statistics, so that I can understand class distribution and identify data quality issues.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL compute and save dataset statistics to dataset_statistics.txt
2. THE dataset statistics SHALL include total number of samples and total number of classes (currently 23, designed for 38-40)
3. THE dataset statistics SHALL include per-class sample counts sorted alphabetically by class name
4. THE dataset statistics SHALL include minimum, maximum, mean, and standard deviation of samples per class
5. THE dataset statistics SHALL note that the current dataset is incomplete and the system is designed for the full Urdu alphabet
6. WHEN a class has fewer than 100 samples, THE Dataset_Loader SHALL include a warning in the statistics file
6. THE Dataset_Loader SHALL print dataset statistics to the console during loading
7. THE dataset statistics file SHALL use UTF-8 encoding to support Urdu alphabet characters

### Requirement 15: Coordinate Validation

**User Story:** As a developer, I want to validate that coordinates are properly normalized, so that I can detect preprocessing errors before training.

#### Acceptance Criteria

1. THE system SHALL provide a validation script that checks coordinate ranges
2. WHEN coordinates are validated, THE validation script SHALL verify all values are within [-2, 2]
3. WHEN coordinates are validated, THE validation script SHALL verify the wrist landmark (first coordinate pair) is near the origin
4. WHEN coordinates are validated, THE validation script SHALL compute and display the bounding box dimensions
5. IF any coordinates are outside the expected range, THE validation script SHALL print a warning with the sample index
6. THE validation script SHALL validate a random sample of 100 coordinates from the training set
7. THE validation script SHALL print a summary of validation results (number of valid samples, number of warnings)

### Requirement 16: Training Progress Visualization

**User Story:** As a machine learning engineer, I want to visualize training progress, so that I can monitor convergence and detect overfitting.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL save epoch number, training loss, validation loss, and validation accuracy to training_log.csv
2. THE training_log.csv file SHALL have a header row with column names
3. THE training_log.csv file SHALL have one row per epoch
4. THE Training_Pipeline SHALL print training and validation metrics to the console after each epoch
5. THE Training_Pipeline SHALL format metric output to 4 decimal places for readability
6. WHEN early stopping triggers, THE Training_Pipeline SHALL indicate which epoch had the best validation loss

### Requirement 17: Inference Performance Requirements

**User Story:** As a user, I want fast inference, so that the system feels responsive during real-time use.

#### Acceptance Criteria

1. THE Inference_Engine SHALL process a single frame in less than 50 milliseconds on a modern CPU
2. THE ANN_Classifier forward pass SHALL complete in less than 5 milliseconds for a single sample
3. THE MediaPipe_Detector SHALL detect hand landmarks in less than 30 milliseconds per frame
4. THE Inference_Engine SHALL maintain a frame rate of at least 15 FPS during webcam demo
5. THE system SHALL run inference on CPU without requiring a GPU

### Requirement 18: Batch Inference Support

**User Story:** As a developer, I want to run inference on batches of samples, so that I can efficiently evaluate the model on the test set.

#### Acceptance Criteria

1. THE ANN_Classifier SHALL accept input tensors of shape (batch_size, 42)
2. THE ANN_Classifier SHALL return output tensors of shape (batch_size, num_classes) where num_classes matches the dataset
3. THE Evaluation_Module SHALL process the entire test set in batches of size 32
4. THE Evaluation_Module SHALL aggregate predictions across all batches for metric computation
5. THE system SHALL support batch sizes from 1 to 256

### Requirement 19: Reproducibility

**User Story:** As a researcher, I want reproducible results, so that I can compare different model configurations reliably.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL set random seeds for NumPy, PyTorch, and Python's random module
2. THE Training_Pipeline SHALL use a fixed random seed value of 42
3. THE Dataset_Loader SHALL use a fixed random seed of 42 for dataset splitting
4. WHEN the same configuration is used, THE Training_Pipeline SHALL produce identical results across multiple runs
5. THE system SHALL document the random seed values in the configuration file

### Requirement 20: Command-Line Interface

**User Story:** As a developer, I want command-line scripts for training, evaluation, and demo, so that I can run the pipeline without modifying code.

#### Acceptance Criteria

1. THE system SHALL provide a train.py script that trains the model when executed
2. THE system SHALL provide an evaluate.py script that evaluates the trained model on the test set
3. THE system SHALL provide a demo.py script that launches the webcam demo
4. THE train.py script SHALL load configuration from config.py automatically
5. THE evaluate.py script SHALL load the trained model from alphabet_classifier.pt automatically
6. THE demo.py script SHALL load the trained model from alphabet_classifier.pt automatically
7. IF the model file does not exist when running evaluate.py or demo.py, THE system SHALL print an error message instructing the user to run train.py first
8. THE system SHALL provide a run_pipeline.py script that executes training, evaluation, and demo in sequence

### Requirement 21: Dataset Extensibility and Future Expansion

**User Story:** As a researcher, I want the system to automatically adapt to expanded datasets, so that I can add new alphabet classes without modifying code.

#### Acceptance Criteria

1. THE system SHALL automatically detect the number of alphabet classes present in the dataset directory
2. WHEN new alphabet folders are added to the dataset, THE Dataset_Loader SHALL include them in training without code changes
3. THE ANN_Classifier SHALL instantiate with the correct output layer size based on the detected number of classes
4. THE system SHALL save the number of classes alongside the model weights for consistent loading
5. WHEN loading a saved model, THE Inference_Engine SHALL verify that the model's class count matches the current dataset
6. IF the model's class count does not match the dataset, THE system SHALL print a warning indicating retraining is required
7. THE label mapping dictionary SHALL be saved alongside the model to maintain consistency between training and inference
8. THE system SHALL support seamless expansion from the current 23 classes to the full 38-40 Urdu alphabet classes
9. THE Dataset_Loader SHALL log the number of detected classes during initialization
10. THE system documentation SHALL note that the current implementation uses 23 classes but is designed for 38-40 classes

