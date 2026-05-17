from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BIDI_CONTROL_PATTERN = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")


@dataclass(frozen=True)
class AlphabetDataset:
    features: np.ndarray
    labels: list[str]
    class_counts: dict[str, int]


def clean_label(label: str) -> str:
    """Normalize Urdu labels and remove invisible direction controls."""
    normalized = unicodedata.normalize("NFC", str(label))
    return BIDI_CONTROL_PATTERN.sub("", normalized).strip()


def normalize_hand_landmarks(flat_xy: np.ndarray) -> np.ndarray:
    """Return a 42-value, wrist-relative, scale-normalized hand vector.

    Input can be pixel coordinates or MediaPipe normalized coordinates. Because
    we subtract the wrist and divide by hand scale, the output is mostly
    invariant to image size and hand distance from the camera.
    """
    pts = np.asarray(flat_xy, dtype=np.float32).reshape(21, 2).copy()

    if not np.any(pts):
        return pts.reshape(-1)

    wrist = pts[0].copy()
    pts -= wrist

    # MediaPipe/OpenPose landmark 9 is the middle-finger MCP. It is stable for
    # scale normalization and matches the old project scripts.
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale

    return pts.reshape(-1).astype(np.float32)


def load_right_hand_dataset(db_path: str | Path) -> AlphabetDataset:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Dataset database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(rightHandDataset)")
        cols = [row[1] for row in cursor.fetchall()]
        if not cols:
            raise ValueError("Table rightHandDataset was not found")

        label_idx = cols.index("label")
        feature_cols = [c for c in cols if c not in ("id", "label")]
        feature_indices = [cols.index(c) for c in feature_cols]

        cursor.execute("SELECT * FROM rightHandDataset")
        rows = cursor.fetchall()
    finally:
        conn.close()

    features: list[np.ndarray] = []
    labels: list[str] = []
    counts: dict[str, int] = {}

    for row in rows:
        raw = np.array([row[i] for i in feature_indices], dtype=np.float32)
        label = clean_label(row[label_idx])
        features.append(normalize_hand_landmarks(raw))
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1

    return AlphabetDataset(
        features=np.asarray(features, dtype=np.float32),
        labels=labels,
        class_counts=dict(sorted(counts.items(), key=lambda item: item[0])),
    )
