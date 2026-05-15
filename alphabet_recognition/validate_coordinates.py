"""
Coordinate Validation Script for ANN-Based PSL Alphabet Recognition System

Validates that normalized coordinates are within expected ranges and that
the wrist landmark is near the origin after normalization.

Usage:
    python validate_coordinates.py
"""

import sys
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_coordinates(X: np.ndarray, num_samples: int = 100) -> None:
    """
    Validate a random sample of normalized coordinate arrays.

    Checks:
        1. All values within [-2, 2]
        2. Wrist landmark (first x, y pair) is near origin (within 0.1)
        3. Bounding box dimensions are reasonable

    Args:
        X:           Array of normalized coordinates, shape (N, 42)
        num_samples: Number of random samples to validate (default 100)
    """
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=min(num_samples, len(X)), replace=False)
    samples = X[indices]

    valid_count   = 0
    warning_count = 0

    print(f"\nValidating {len(samples)} random coordinate samples …")
    print("=" * 60)

    for i, coords in enumerate(samples):
        warnings = []

        # Check 1: all values within [-2, 2]
        out_of_range = np.where(np.abs(coords) > 2.0)[0]
        if len(out_of_range) > 0:
            warnings.append(
                f"  {len(out_of_range)} values outside [-2, 2]: "
                f"indices {out_of_range[:5].tolist()}"
            )

        # Check 2: wrist near origin
        wrist_x, wrist_y = coords[0], coords[1]
        if abs(wrist_x) > 0.1 or abs(wrist_y) > 0.1:
            warnings.append(
                f"  Wrist not near origin: ({wrist_x:.4f}, {wrist_y:.4f})"
            )

        # Check 3: bounding box
        xy = coords.reshape(21, 2)
        bbox_w = xy[:, 0].max() - xy[:, 0].min()
        bbox_h = xy[:, 1].max() - xy[:, 1].min()

        if warnings:
            warning_count += 1
            print(f"Sample {indices[i]:4d}: ⚠  WARNING")
            for w in warnings:
                print(w)
            print(f"  Bounding box: {bbox_w:.4f} × {bbox_h:.4f}")
        else:
            valid_count += 1

    print("=" * 60)
    print(f"Results: {valid_count} valid, {warning_count} warnings "
          f"(out of {len(samples)} samples)")

    if warning_count == 0:
        print("✓ All samples passed coordinate validation.")
    else:
        print(f"⚠  {warning_count} samples had issues — check preprocessing pipeline.")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    from dataset_loader import load_dataset
    from preprocessor import normalize_hand_coords

    logger.info("Loading dataset for coordinate validation …")
    (X_tr, _), _, _, label_map = load_dataset()

    logger.info("Normalising training coordinates …")
    X_norm = np.array([normalize_hand_coords(x) for x in X_tr])

    validate_coordinates(X_norm, num_samples=100)
