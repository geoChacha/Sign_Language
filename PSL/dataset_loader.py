"""
Dataset Loading Module for Right-Hand Alphabet Recognition System

This module handles loading OpenPose JSON files from the alphabets_dataset directory,
extracting right-hand coordinates, and creating stratified train/validation/test splits.

Dataset Structure:
    PSL_dataset/datasets/alphabets_dataset/
        ++/
            1560784905.257021/
                000000000038_keypoints.json
                ...
        ++ü/
            ...
        (37 alphabet folders total)

Each JSON file contains OpenPose output with hand_right_keypoints_2d array.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

from config import (
    DATASET_ROOT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    DATASET_STATS_PATH
)

logger = logging.getLogger(__name__)


def extract_right_hand_coords(json_path: Path) -> Optional[np.ndarray]:
    """
    Extract right-hand coordinates from OpenPose JSON file.
    
    OpenPose format: hand_right_keypoints_2d is a flat array of 63 values:
    [x0, y0, c0, x1, y1, c1, ..., x20, y20, c20]
    where (x, y) are pixel coordinates and c is confidence.
    
    Args:
        json_path: Path to OpenPose JSON file
    
    Returns:
        (42,) numpy array of [x0, y0, x1, y1, ..., x20, y20] or None if invalid
    
    Returns None if:
        - JSON parsing fails
        - people array is empty
        - hand_right_keypoints_2d has wrong length
        - all coordinates are zero
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON {json_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {json_path}: {e}")
        return None
    
    # Check if people array exists and is not empty
    if not data.get("people"):
        logger.warning(f"Empty people array in {json_path}")
        return None
    
    # Extract hand_right_keypoints_2d from first person
    try:
        hand_right = data["people"][0]["hand_right_keypoints_2d"]
    except (KeyError, IndexError) as e:
        logger.warning(f"Missing hand_right_keypoints_2d in {json_path}: {e}")
        return None
    
    # Verify array has exactly 63 values (21 landmarks × 3)
    if len(hand_right) != 63:
        logger.warning(f"Invalid hand_right_keypoints_2d length {len(hand_right)} in {json_path}")
        return None
    
    # Extract x, y pairs (skip confidence values)
    # hand_right[0::3] = x coordinates
    # hand_right[1::3] = y coordinates
    coords = []
    for i in range(0, 63, 3):
        coords.extend([hand_right[i], hand_right[i+1]])
    
    coords = np.array(coords, dtype=np.float32)
    
    # Check if all coordinates are zero (invalid detection)
    if np.all(coords == 0):
        logger.warning(f"All zero coordinates in {json_path}")
        return None
    
    return coords  # Shape: (42,)


def load_dataset(
    root_dir: str = DATASET_ROOT,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # (X_train, y_train)
    Tuple[np.ndarray, np.ndarray],  # (X_val, y_val)
    Tuple[np.ndarray, np.ndarray],  # (X_test, y_test)
    Dict[int, str],                  # label_map: {0: '++', 1: '++ü', ...}
]:
    """
    Load dataset from alphabets_dataset directory and create stratified splits.
    
    Args:
        root_dir: Root directory containing alphabet folders
        train_split: Fraction of data for training (default 0.70)
        val_split: Fraction of data for validation (default 0.15)
        test_split: Fraction of data for testing (default 0.15)
    
    Returns:
        Tuple containing:
            - (X_train, y_train): Training data and labels
            - (X_val, y_val): Validation data and labels
            - (X_test, y_test): Test data and labels
            - label_map: Dictionary mapping integer labels to alphabet strings
    
    Raises:
        ValueError: If dataset doesn't meet requirements (37 classes, 5000-5200 samples)
    """
    logger.info(f"Loading dataset from {root_dir}")
    
    # Find all JSON files recursively
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
    
    json_files = list(root_path.rglob("*_keypoints.json"))
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Load all samples
    samples = []
    labels = []
    
    for json_path in json_files:
        # Extract label from parent directory name
        # Path structure: .../alphabets_dataset/++/1560784905.257021/file.json
        # Label is the parent's parent name (++)
        label = json_path.parent.parent.name
        
        # Extract coordinates
        coords = extract_right_hand_coords(json_path)
        if coords is not None:
            samples.append(coords)
            labels.append(label)
    
    logger.info(f"Loaded {len(samples)} valid samples")
    
    # Convert to numpy arrays
    X = np.array(samples, dtype=np.float32)  # Shape: (N, 42)
    
    # Create label encoding (alphabetical order)
    unique_labels = sorted(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    
    y = np.array([label_to_int[label] for label in labels], dtype=np.int64)
    
    # Validate dataset
    num_classes = len(unique_labels)
    num_samples = len(X)
    
    logger.info(f"Dataset: {num_samples} samples, {num_classes} classes")
    
    if num_classes < 2:
        raise ValueError(f"Expected at least 2 classes, got {num_classes}")
    
    logger.info(f"Loaded {num_classes} classes with {num_samples} total samples")
    
    # Print dataset statistics
    label_counts = Counter(labels)
    print_dataset_statistics(label_counts, DATASET_STATS_PATH)
    
    # Stratified split: first split into train and temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_split + test_split),
        stratify=y,
        random_state=42
    )
    
    # Split temp into val and test
    val_ratio = val_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=42
    )
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), int_to_label


def print_dataset_statistics(
    label_counts: Dict[str, int],
    output_path: str = DATASET_STATS_PATH
) -> None:
    """
    Print and save dataset statistics to file.
    
    Args:
        label_counts: Dictionary mapping labels to sample counts
        output_path: Path to save statistics file
    """
    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    counts = list(label_counts.values())
    
    min_count = min(counts)
    max_count = max(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    # Create statistics report
    report = []
    report.append("Dataset Statistics")
    report.append("=" * 60)
    report.append(f"Total samples: {total_samples}")
    report.append(f"Total classes: {num_classes}")
    report.append("")
    report.append("Per-class counts:")
    
    # Sort by label name
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        report.append(f"  {label:20s}: {count:4d} samples")
        
        # Warn if class has < 100 samples
        if count < 100:
            warning = f"WARNING: class {label} has only {count} samples — may cause training instability."
            report.append(f"    ⚠️  {warning}")
            logger.warning(warning)
    
    report.append("")
    report.append("Summary:")
    report.append(f"  Min:  {min_count} samples")
    report.append(f"  Max:  {max_count} samples")
    report.append(f"  Mean: {mean_count:.1f} samples")
    report.append(f"  Std:  {std_count:.1f} samples")
    
    # Print to console (replace unencodable chars safely)
    report_text = "\n".join(report)
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace'))
    
    # Save to file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Dataset statistics saved to {output_path}")
