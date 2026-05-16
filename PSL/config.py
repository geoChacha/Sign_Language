"""
Configuration Module for ANN-Based PSL Alphabet Recognition System

This module serves as the single source of truth for all system constants,
hyperparameters, and file paths. All other modules import from this configuration
to ensure consistency across the entire pipeline.

Architecture: Feedforward neural network (42 → 128 → 64 → num_classes)
Dataset: All Urdu alphabet classes in PSL_dataset (dynamically detected)
Input: 21 hand landmarks × 2 coordinates = 42 dimensions
"""

from pathlib import Path

# Project root is one level up from this config file (alphabet_recognition/../)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Configuration
DATASET_ROOT = str(_PROJECT_ROOT / "PSL_dataset" / "datasets" / "alphabets_dataset")
# Note: NUM_CLASSES is determined dynamically from the dataset directory
# The output layer size will be set based on the number of alphabet folders found
INPUT_DIM = 42     # 21 landmarks × 2 coordinates (x, y)

# Model Architecture
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT = 0.3

# Training Hyperparameters
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42  # Fixed seed for reproducibility
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Data Augmentation Parameters
AUGMENT_NOISE_STD = 0.02  # Standard deviation for Gaussian noise
AUGMENT_ROTATION_DEG = 15.0  # Maximum rotation angle in degrees (±15°)
AUGMENT_SCALE_RANGE = (0.9, 1.1)  # Uniform scaling factor range

# Inference Configuration
CONFIDENCE_THRESHOLD = 0.70  # Minimum probability for confident prediction

# Output File Paths (anchored to alphabet_recognition/ directory)
_MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(_MODULE_DIR / "alphabet_classifier.pt")
TRAINING_LOG_PATH = str(_MODULE_DIR / "training_log.csv")
EVAL_REPORT_PATH = str(_MODULE_DIR / "evaluation_report.txt")
CONFUSION_MATRIX_TXT = str(_MODULE_DIR / "confusion_matrix.txt")
CONFUSION_MATRIX_PNG = str(_MODULE_DIR / "confusion_matrix.png")
DATASET_STATS_PATH = str(_MODULE_DIR / "dataset_statistics.txt")
