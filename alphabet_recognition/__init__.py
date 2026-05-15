"""
Right-Hand Alphabet Recognition System

A Pakistan Sign Language (PSL) alphabet recognition system for 37 Urdu alphabets
using static right-hand poses.

Architecture:
    - Input: 21 right-hand landmarks × 2 coordinates = 42 dimensions
    - Model: Feedforward neural network (42 → 128 → 64 → 37)
    - Dataset: 5,112 samples from rightHandDataset
    - Real-time inference: MediaPipe Hands + trained classifier

Modules:
    - config: System constants and hyperparameters
    - preprocessor: scalePoints normalization
    - dataset_loader: OpenPose JSON loading and stratified splitting
    - model: AlphabetClassifier neural network
    - train: Training loop with augmentation and early stopping
    - evaluate: Evaluation metrics and confusion matrix
    - demo: Real-time webcam demo
    - validate_coordinates: MediaPipe/OpenPose compatibility validation
    - run_pipeline: Unified entry point

Usage:
    python alphabet_recognition/run_pipeline.py --mode all
"""

__version__ = "1.0.0"
__author__ = "PSL Recognition Team"

from alphabet_recognition.config import *
from alphabet_recognition.model import AlphabetClassifier
from alphabet_recognition.preprocessor import normalize_hand_coords
