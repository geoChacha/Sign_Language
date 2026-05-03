"""
Pytest configuration and shared fixtures for WLASL-100 model tests.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def dummy_keypoints():
    """Generate dummy keypoint sequence (100 frames, 126 features)."""
    return np.random.randn(100, 126).astype(np.float32)


@pytest.fixture
def dummy_keypoints_64():
    """Generate dummy keypoint sequence (64 frames, 126 features)."""
    return np.random.randn(64, 126).astype(np.float32)


@pytest.fixture
def dummy_frame():
    """Generate dummy video frame (480x640x3)."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def clear_background_frame():
    """Generate frame with clear background (solid color)."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
    return frame


@pytest.fixture
def cluttered_background_frame():
    """Generate frame with cluttered background (random noise)."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def dummy_video_file(tmp_path):
    """Create a dummy video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    
    # Create a simple video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Write 90 frames (3 seconds at 30fps)
    for i in range(90):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    return video_path


@pytest.fixture
def corrupted_video_file(tmp_path):
    """Create a corrupted video file for error testing."""
    video_path = tmp_path / "corrupted.mp4"
    
    # Write random bytes to simulate corrupted file
    with open(video_path, 'wb') as f:
        f.write(b'corrupted video data')
    
    return video_path


@pytest.fixture
def model_path():
    """Return path to model checkpoint."""
    return "app/ml/models/wlasl100/best_model.pth"


@pytest.fixture
def vocab_path():
    """Return path to vocabulary file."""
    return "app/ml/models/wlasl100/vocab.json"


@pytest.fixture
def temperature_path():
    """Return path to temperature file."""
    return "app/ml/models/wlasl100/temperature.json"


@pytest.fixture
def sample_vocab():
    """Return sample vocabulary dictionary."""
    return {str(i): f"sign_{i}" for i in range(100)}


@pytest.fixture
def sample_temperature():
    """Return sample temperature value."""
    return {"temperature": 1.0}
