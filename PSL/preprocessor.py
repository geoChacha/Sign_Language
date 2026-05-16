"""
Preprocessing Module for Right-Hand Alphabet Recognition System

This module implements the scalePoints normalization algorithm for hand coordinates.
The normalization provides translation and scale invariance, ensuring the model
generalizes across different hand positions and sizes.

Algorithm:
1. Compute centroid of all landmarks
2. Translate all points to origin (subtract centroid)
3. Compute bounding box dimensions
4. Scale by maximum bounding box dimension

Properties:
- Translation-invariant: normalize(coords + offset) == normalize(coords)
- Scale-invariant: normalize(coords * scale) == normalize(coords) for scale > 0
- Rotation-variant: preserves hand orientation (intentional)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_hand_coords(coords: np.ndarray) -> np.ndarray:
    """
    Apply scalePoints normalization to right-hand coordinates.
    
    This function normalizes hand coordinates to be translation and scale invariant,
    making the model robust to different hand positions and sizes in the frame.
    
    Args:
        coords: Hand coordinates as either:
                - (42,) flat array: [x0, y0, x1, y1, ..., x20, y20]
                - (21, 2) array: [[x0, y0], [x1, y1], ..., [x20, y20]]
    
    Returns:
        Normalized coordinates as (42,) flat array.
        Returns zero vector if all landmarks are identical (degenerate case).
    
    Examples:
        >>> coords = np.array([100, 200, 110, 210, 120, 220])  # 3 landmarks
        >>> normalized = normalize_hand_coords(coords)
        >>> normalized.shape
        (6,)
    """
    # Reshape to (21, 2) if input is flat
    original_shape = coords.shape
    if coords.shape == (42,):
        coords = coords.reshape(21, 2)
    elif coords.shape != (21, 2):
        raise ValueError(f"Expected shape (42,) or (21, 2), got {original_shape}")
    
    # Compute centroid (mean of all x, y coordinates)
    centroid = coords.mean(axis=0)  # Shape: (2,)
    
    # Translate to origin
    translated = coords - centroid
    
    # Compute bounding box
    min_xy = translated.min(axis=0)
    max_xy = translated.max(axis=0)
    bbox_size = max_xy - min_xy  # (width, height)
    
    # Scale by maximum dimension
    max_dim = bbox_size.max()
    
    # Handle degenerate case: all landmarks identical
    if max_dim == 0 or np.isclose(max_dim, 0):
        logger.warning("Degenerate hand pose detected: all landmarks identical")
        return np.zeros(42, dtype=np.float32)
    
    # Scale coordinates
    scaled = translated / max_dim
    
    # Return as flat (42,) array
    return scaled.flatten().astype(np.float32)


def add_gaussian_noise(coords: np.ndarray, std: float = 0.02) -> np.ndarray:
    """
    Add Gaussian noise to coordinates for data augmentation.
    
    Args:
        coords: Hand coordinates, shape (42,) or (21, 2)
        std: Standard deviation of Gaussian noise (default 0.02)
    
    Returns:
        Noisy coordinates with same shape as input
    """
    noise = np.random.normal(0, std, coords.shape)
    return coords + noise.astype(np.float32)


def rotate_coordinates(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate coordinates around the wrist (first landmark) for data augmentation.
    
    Args:
        coords: Hand coordinates, shape (42,) or (21, 2)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
    
    Returns:
        Rotated coordinates with same shape as input
    """
    # Reshape to (21, 2) if flat
    original_shape = coords.shape
    if coords.shape == (42,):
        coords = coords.reshape(21, 2)
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_degrees)
    
    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float32)
    
    # Wrist is at index 0 - use as center of rotation
    wrist = coords[0].copy()
    
    # Translate to origin (wrist at 0,0)
    translated = coords - wrist
    
    # Apply rotation
    rotated = translated @ rotation_matrix.T
    
    # Translate back
    rotated = rotated + wrist
    
    # Return in original shape
    if original_shape == (42,):
        return rotated.flatten().astype(np.float32)
    return rotated.astype(np.float32)


def scale_coordinates(coords: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale coordinates uniformly for data augmentation.
    
    Args:
        coords: Hand coordinates, shape (42,) or (21, 2)
        scale_factor: Uniform scaling factor (e.g., 1.1 = 10% larger)
    
    Returns:
        Scaled coordinates with same shape as input
    """
    # Reshape to (21, 2) if flat
    original_shape = coords.shape
    if coords.shape == (42,):
        coords = coords.reshape(21, 2)
    
    # Wrist is at index 0 - use as center of scaling
    wrist = coords[0].copy()
    
    # Translate to origin
    translated = coords - wrist
    
    # Apply scaling
    scaled = translated * scale_factor
    
    # Translate back
    scaled = scaled + wrist
    
    # Return in original shape
    if original_shape == (42,):
        return scaled.flatten().astype(np.float32)
    return scaled.astype(np.float32)


def augment_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Apply random augmentation transformations to coordinates.
    
    Applies the following transformations in random order:
    - Gaussian noise with std=0.02
    - Random rotation in [-15°, +15°]
    - Random scaling in [0.9, 1.1]
    
    Args:
        coords: Hand coordinates, shape (42,) or (21, 2)
    
    Returns:
        Augmented coordinates with same shape as input
    """
    from config import AUGMENT_NOISE_STD, AUGMENT_ROTATION_DEG, AUGMENT_SCALE_RANGE
    
    # Create list of transformations
    transformations = [
        lambda c: add_gaussian_noise(c, std=AUGMENT_NOISE_STD),
        lambda c: rotate_coordinates(c, angle_degrees=np.random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG)),
        lambda c: scale_coordinates(c, scale_factor=np.random.uniform(*AUGMENT_SCALE_RANGE))
    ]
    
    # Shuffle transformations for random order
    np.random.shuffle(transformations)
    
    # Apply transformations sequentially
    augmented = coords.copy()
    for transform in transformations:
        augmented = transform(augmented)
    
    return augmented.astype(np.float32)
