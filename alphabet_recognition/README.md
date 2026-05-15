# ANN-Based PSL Alphabet Recognition System

A feedforward Artificial Neural Network (ANN) system for recognizing Pakistan Sign Language (PSL) alphabet hand poses using 2D hand keypoint coordinates.

## Overview

This system classifies static hand gestures representing Urdu alphabet characters using:
- **Training**: OpenPose hand keypoints from JSON files
- **Inference**: Real-time MediaPipe hand detection via webcam
- **Architecture**: Feedforward neural network (42 → 128 → 64 → num_classes)
- **Dynamic class detection**: Automatically adapts to all alphabet folders in the dataset

## Features

- ✅ **Dynamic class count**: Automatically detects and supports all alphabet classes in the dataset
- ✅ **Translation & scale invariant**: Normalization ensures robustness to hand position and size
- ✅ **Data augmentation**: Gaussian noise, rotation, and scaling during training
- ✅ **Early stopping**: Prevents overfitting with patience-based validation monitoring
- ✅ **Real-time inference**: MediaPipe-powered webcam demo with confidence display
- ✅ **Comprehensive evaluation**: Per-class metrics, confusion matrix, and problematic class identification

## System Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux
- Webcam (for real-time demo)
- CPU sufficient (GPU optional)

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd alphabet_recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The system expects OpenPose JSON files organized as follows:

```
PSL_dataset/datasets/alphabets_dataset/
├── alphabet_1/
│   ├── session_1/
│   │   ├── 000000000001_keypoints.json
│   │   ├── 000000000002_keypoints.json
│   │   └── ...
│   └── session_2/
│       └── ...
├── alphabet_2/
│   └── ...
└── ...
```

Each JSON file contains OpenPose output with `hand_right_keypoints_2d` array (21 landmarks × 3 values: x, y, confidence).

## Usage

### Training

Train the model on your dataset:

```bash
python run_pipeline.py --mode train
```

This will:
- Load and normalize all alphabet classes from the dataset
- Train the model with data augmentation and early stopping
- Save the best model to `alphabet_classifier.pt`
- Generate training log in `training_log.csv`

### Evaluation

Evaluate the trained model on the test set:

```bash
python run_pipeline.py --mode evaluate
```

This will:
- Load the trained model
- Compute accuracy, precision, recall, F1-score per class
- Generate confusion matrix (text + PNG heatmap)
- Save evaluation report to `evaluation_report.txt`

### Real-Time Demo

Run the webcam demo:

```bash
python run_pipeline.py --mode demo
```

Or with custom confidence threshold:

```bash
python run_pipeline.py --mode demo --threshold 0.75
```

Controls:
- Press **'q'** to quit

The demo displays:
- Predicted alphabet label (green for high confidence ≥0.85, yellow for medium 0.70-0.84)
- Confidence percentage
- Hand skeleton overlay
- FPS counter

### Full Pipeline

Run training, evaluation, and demo in sequence:

```bash
python run_pipeline.py --mode all
```

## Configuration

All hyperparameters and paths are centralized in `config.py`:

```python
# Model architecture
INPUT_DIM = 42          # 21 landmarks × 2 coordinates
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT = 0.3

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Data augmentation
AUGMENT_NOISE_STD = 0.02
AUGMENT_ROTATION_DEG = 15.0
AUGMENT_SCALE_RANGE = (0.9, 1.1)

# Inference
CONFIDENCE_THRESHOLD = 0.70
```

## Model Architecture

```
Input (42 dimensions: 21 landmarks × 2 coords)
    ↓
Linear(42 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(64 → num_classes)
    ↓
Output (logits for all detected alphabet classes)
```

**Parameter count**: ~15,000 (varies with number of classes)

## Coordinate Normalization

The system applies translation and scale invariant normalization:

1. Compute centroid of all 21 landmarks
2. Translate coordinates to center at origin
3. Compute bounding box dimensions
4. Scale by maximum bounding box dimension

This ensures the model generalizes across different hand positions, sizes, and distances from the camera.

## Output Files

After training and evaluation, the following files are generated:

- `alphabet_classifier.pt` — Trained model checkpoint (includes weights, num_classes, label_map)
- `training_log.csv` — Per-epoch training and validation metrics
- `dataset_statistics.txt` — Dataset statistics and per-class sample counts
- `evaluation_report.txt` — Comprehensive evaluation metrics
- `confusion_matrix.txt` — Confusion matrix in text format
- `confusion_matrix.png` — Confusion matrix heatmap visualization

## Validation

Validate coordinate normalization:

```bash
python validate_coordinates.py
```

This checks that:
- All normalized values are within [-2, 2]
- Wrist landmark is near the origin
- Bounding box dimensions are reasonable

## Extensibility

The system automatically adapts to the number of alphabet classes in your dataset:

- **Adding new classes**: Simply add new alphabet folders to the dataset directory — no code changes needed
- **Model retraining**: Required when the number of classes changes
- **Label consistency**: The label map is saved with the model to ensure consistent predictions

## Troubleshooting

**Model file not found**:
- Run `python run_pipeline.py --mode train` first to train the model

**Low accuracy (<60%)**:
- Check dataset quality (ensure valid hand detections)
- Increase training data per class (aim for 100+ samples per class)
- Adjust hyperparameters in `config.py`

**Webcam not opening**:
- Ensure no other application is using the webcam
- Check camera permissions
- Try different camera index: modify `camera_index` parameter in `demo.py`

**Unicode encoding errors on Windows**:
- The system handles UTF-8 encoding automatically
- If issues persist, ensure your terminal supports UTF-8

## Performance

- **Training time**: ~2-5 minutes for 1,700 samples on CPU
- **Inference speed**: 15-30 FPS on modern CPU
- **Model size**: ~60 KB (compressed checkpoint)

## License

This project is part of the PSL alphabet recognition research.

## Citation

If you use this system in your research, please cite:

```
ANN-Based PSL Alphabet Recognition System
Pakistan Sign Language Alphabet Classification using Hand Keypoints
```

## Contact

For questions or issues, please refer to the project documentation or contact the development team.
