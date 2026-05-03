"""
keypoint_extractor.py - MediaPipe Holistic keypoint extraction for ASL recognition.

Extracts hand landmarks (21 points per hand × 3 coordinates = 63 features per hand)
from video frames using MediaPipe Holistic with CLAHE preprocessing.
"""

import cv2
import numpy as np
from typing import Optional
import logging

try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available or incompatible version")


logger = logging.getLogger(__name__)


class KeypointExtractor:
    """
    Extracts hand keypoints from video frames using MediaPipe Holistic.
    
    Features extracted per frame:
    - Left hand: 21 landmarks × 3 (x, y, z) = 63 features
    - Right hand: 21 landmarks × 3 (x, y, z) = 63 features
    - Total: 126 features per frame
    
    Note: Pose landmarks are NOT extracted as they depend on camera framing
    and reduce generalization. Hands-only features are framing-invariant.
    """
    
    def __init__(self, model_complexity: int = 2):
        """
        Initialize MediaPipe Holistic extractor.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
                             2 = best accuracy (recommended for production)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not available. Install with: pip install mediapipe==0.9.3"
            )
        
        self.model_complexity = model_complexity
        self.holistic = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # CLAHE for contrast normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        logger.info(f"KeypointExtractor initialized with model_complexity={model_complexity}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast normalization in LAB color space.
        
        This makes phone-camera frames look closer to studio-lit WLASL recordings,
        improving MediaPipe keypoint quality on real-world input.
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            Preprocessed BGR image (H, W, 3)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge and convert back to BGR
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def landmarks_to_array(self, landmarks, n_points: int) -> np.ndarray:
        """
        Convert MediaPipe landmarks to flat numpy array.
        
        Args:
            landmarks: MediaPipe landmark list or None
            n_points: Expected number of landmarks
        Returns:
            Flat array of shape (n_points * 3,) with x, y, z coordinates
            Returns zeros if landmarks is None
        """
        if landmarks is None:
            return np.zeros(n_points * 3, dtype=np.float32)
        
        arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32
        )
        return arr.flatten()
    
    def extract_keypoints_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract hand keypoints from a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            Keypoint array of shape (126,) containing:
            - [0:63]: Left hand landmarks (21 × 3)
            - [63:126]: Right hand landmarks (21 × 3)
            Returns zero-filled array if hands are not detected
        """
        # Preprocess frame
        frame = self.preprocess_frame(frame)
        
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        try:
            result = self.holistic.process(rgb)
        except Exception as e:
            logger.warning(f"MediaPipe processing failed: {e}")
            return np.zeros(126, dtype=np.float32)
        
        # Extract hand landmarks
        left_hand = self.landmarks_to_array(result.left_hand_landmarks, 21)
        right_hand = self.landmarks_to_array(result.right_hand_landmarks, 21)
        
        # Concatenate: [left_hand(63) + right_hand(63)] = 126 features
        keypoints = np.concatenate([left_hand, right_hand])
        
        return keypoints
    
    def extract_keypoints_from_frames(
        self,
        frames: list[np.ndarray]
    ) -> np.ndarray:
        """
        Extract keypoints from multiple frames.
        
        Args:
            frames: List of BGR images
        Returns:
            Keypoint sequence of shape (num_frames, 126)
        """
        keypoints_list = []
        
        for i, frame in enumerate(frames):
            try:
                kp = self.extract_keypoints_from_frame(frame)
                keypoints_list.append(kp)
            except Exception as e:
                logger.warning(f"Failed to extract keypoints from frame {i}: {e}")
                # Use zero-filled features for failed frames
                keypoints_list.append(np.zeros(126, dtype=np.float32))
        
        return np.array(keypoints_list, dtype=np.float32)
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()
            logger.info("KeypointExtractor closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


def create_extractor(model_complexity: int = 2) -> KeypointExtractor:
    """
    Factory function to create a KeypointExtractor instance.
    
    Args:
        model_complexity: MediaPipe model complexity (0, 1, or 2)
    Returns:
        KeypointExtractor instance
    """
    return KeypointExtractor(model_complexity=model_complexity)


if __name__ == "__main__":
    # Test keypoint extraction
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Testing KeypointExtractor...")
    with create_extractor() as extractor:
        keypoints = extractor.extract_keypoints_from_frame(dummy_frame)
        print(f"Extracted keypoints shape: {keypoints.shape}")
        print(f"Expected shape: (126,)")
        print(f"Left hand features: {keypoints[:63].shape}")
        print(f"Right hand features: {keypoints[63:126].shape}")
        
        # Test with multiple frames
        frames = [dummy_frame] * 10
        sequence = extractor.extract_keypoints_from_frames(frames)
        print(f"\nExtracted sequence shape: {sequence.shape}")
        print(f"Expected shape: (10, 126)")
    
    print("\n✓ KeypointExtractor test completed")
