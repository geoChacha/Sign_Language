"""
video_validator.py — Sign Language Video Validation Service

Validates that uploaded videos contain actual sign language content by:
1. Checking for hand/body landmarks using MediaPipe
2. Verifying video duration is within acceptable range
3. Ensuring minimum motion/activity in the video
4. Validating video quality (resolution, FPS)

Used by sign-to-text endpoints to reject invalid/irrelevant videos.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class VideoValidationError(Exception):
    """Raised when video fails sign language validation checks."""
    pass


class SignLanguageVideoValidator:
    """
    Validates that a video contains sign language content.
    
    Checks performed:
    - Hand landmarks detected in at least N% of frames
    - Video duration within acceptable range (2-30 seconds)
    - Minimum motion detected (not a static image)
    - Resolution adequate for landmark detection (min 240p)
    """
    
    def __init__(
        self,
        min_duration_sec: float = 0.5,
        max_duration_sec: float = 30.0,
        min_hand_frames_ratio: float = 0.30,  # At least 30% of frames must have hands
        min_resolution: Tuple[int, int] = (320, 240),
        sample_frame_count: int = 30,  # Sample frames uniformly
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_hand_frames_ratio = min_hand_frames_ratio
        self.min_resolution = min_resolution
        self.sample_frame_count = sample_frame_count
        
        # MediaPipe Hands instance (created on-demand)
        self._hands = None
    
    def _get_hands_detector(self):
        """Lazy-load MediaPipe Hands detector."""
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._hands
    
    def validate_video_file(self, video_path: str) -> Dict:
        """
        Validate a video file for sign language content.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with validation results and metrics
            
        Raises:
            VideoValidationError: If video fails validation checks
        """
        if not os.path.exists(video_path):
            raise VideoValidationError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoValidationError("Cannot open video file — file may be corrupted")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames == 0 or fps == 0:
                raise VideoValidationError("Invalid video: no frames or invalid FPS")
            
            duration_sec = total_frames / fps
            
            # Check 1: Duration within acceptable range
            if duration_sec < self.min_duration_sec:
                raise VideoValidationError(
                    f"Video too short ({duration_sec:.1f}s). "
                    f"Minimum duration for sign language: {self.min_duration_sec}s"
                )
            
            if duration_sec > self.max_duration_sec:
                raise VideoValidationError(
                    f"Video too long ({duration_sec:.1f}s). "
                    f"Maximum duration: {self.max_duration_sec}s"
                )
            
            # Check 2: Resolution adequate for landmark detection
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                raise VideoValidationError(
                    f"Video resolution too low ({width}×{height}). "
                    f"Minimum: {self.min_resolution[0]}×{self.min_resolution[1]}"
                )
            
            # Check 3: Hand landmarks detected in sufficient frames
            hands_detector = self._get_hands_detector()
            
            # Sample frames uniformly across the video
            sample_indices = np.linspace(0, total_frames - 1, 
                                        min(self.sample_frame_count, total_frames), 
                                        dtype=int)
            
            frames_with_hands = 0
            frames_sampled = 0
            motion_detected = False
            prev_gray = None
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frames_sampled += 1
                
                # Detect hands
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands_detector.process(rgb)
                
                if result.multi_hand_landmarks:
                    frames_with_hands += 1
                
                # Check for motion (prevent static image uploads)
                if not motion_detected:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        motion_score = np.mean(diff)
                        if motion_score > 5.0:  # Threshold for meaningful motion
                            motion_detected = True
                    prev_gray = gray
            
            if frames_sampled == 0:
                raise VideoValidationError("Could not read any frames from video")
            
            hands_ratio = frames_with_hands / frames_sampled
            
            # Check 4: Sufficient hand visibility
            if hands_ratio < self.min_hand_frames_ratio:
                raise VideoValidationError(
                    f"Insufficient hand visibility detected. "
                    f"Only {hands_ratio*100:.0f}% of frames contain visible hands. "
                    f"For sign language videos, at least {self.min_hand_frames_ratio*100:.0f}% "
                    f"of frames should show hands clearly. "
                    f"Please ensure:\n"
                    f"  • Your hands are fully visible in frame\n"
                    f"  • Good lighting (hands clearly visible)\n"
                    f"  • Camera at chest/torso level\n"
                    f"  • Plain background (avoid cluttered backgrounds)"
                )
            
            # Check 5: Motion detected (not a static image)
            if not motion_detected:
                raise VideoValidationError(
                    "No motion detected — video appears to be a static image. "
                    "Sign language videos must show hand/body movement."
                )
            
            return {
                "valid": True,
                "duration_sec": round(duration_sec, 2),
                "resolution": f"{width}×{height}",
                "fps": round(fps, 1),
                "total_frames": total_frames,
                "frames_sampled": frames_sampled,
                "frames_with_hands": frames_with_hands,
                "hand_visibility_ratio": round(hands_ratio, 3),
                "motion_detected": motion_detected,
            }
        
        finally:
            cap.release()
            if self._hands:
                self._hands.close()
                self._hands = None
    
    def validate_from_bytes(self, video_bytes: bytes, original_filename: str = "video") -> Dict:
        """
        Validate video content from bytes (e.g. uploaded file).
        
        Args:
            video_bytes: Raw video file bytes
            original_filename: Original filename (for extension detection)
            
        Returns:
            Dict with validation results
            
        Raises:
            VideoValidationError: If video fails validation
        """
        # Write to temporary file
        ext = Path(original_filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            return self.validate_video_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass


# Singleton instance
_validator: Optional[SignLanguageVideoValidator] = None


def get_validator() -> SignLanguageVideoValidator:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        _validator = SignLanguageVideoValidator(
            min_duration_sec=0.5,
            max_duration_sec=30.0,
            min_hand_frames_ratio=0.30,  # 30% of frames must show hands
            sample_frame_count=30,
        )
    return _validator


def validate_sign_video(video_path: str) -> Dict:
    """
    Convenience function to validate a sign language video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict with validation metrics
        
    Raises:
        VideoValidationError: If validation fails
    """
    return get_validator().validate_video_file(video_path)


def validate_sign_video_bytes(video_bytes: bytes, filename: str = "video.mp4") -> Dict:
    """
    Convenience function to validate sign video from bytes.
    
    Args:
        video_bytes: Raw video bytes
        filename: Original filename
        
    Returns:
        Dict with validation metrics
        
    Raises:
        VideoValidationError: If validation fails
    """
    return get_validator().validate_from_bytes(video_bytes, filename)
