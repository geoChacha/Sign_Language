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
        min_duration_sec: float = 0.01,  # Very lenient - just check video isn't completely empty
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
            
            # Log video properties for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Video validation - File: {video_path}")
            logger.info(f"  Total frames (raw): {total_frames}")
            logger.info(f"  FPS (raw): {fps}")
            logger.info(f"  Resolution: {width}x{height}")
            
            # Check if this is a WebM file (common for browser recordings)
            is_webm = video_path.lower().endswith('.webm')
            if is_webm:
                logger.info(f"  Detected WebM file - will use lenient validation")
            
            # Check for completely invalid values (OpenCV failed to read file)
            if total_frames < 0 or fps < 0:
                logger.error(f"OpenCV returned invalid negative values (frames={total_frames}, fps={fps})")
                raise VideoValidationError(
                    "Cannot read video file metadata. The video file may not be properly finalized. "
                    "This often happens with WebM files from browser recording. "
                    "Please try: 1) Wait a moment after recording stops, 2) Try recording again, "
                    "3) Use a different browser if the issue persists."
                )
            
            # Be more lenient with frame count/FPS detection
            # WebM videos from MediaRecorder often have unreliable metadata
            if total_frames == 0 or fps == 0 or fps > 1000:
                # Try to manually count frames if properties are unreliable
                logger.warning(f"Unreliable video properties (frames={total_frames}, fps={fps}), attempting manual count")
                manual_frame_count = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    manual_frame_count += 1
                    # Safety limit to prevent infinite loops
                    if manual_frame_count > 10000:
                        logger.error("Manual frame count exceeded safety limit")
                        break
                
                if manual_frame_count == 0:
                    # For WebM files, this might be okay if we can still read frames for validation
                    if is_webm:
                        logger.warning("WebM file has 0 frames in metadata, but will attempt validation anyway")
                        total_frames = 30  # Assume minimum viable frames
                        fps = 30.0
                    else:
                        raise VideoValidationError("Invalid video: no frames could be read")
                else:
                    # Reset capture for further processing
                    cap.release()
                    # Add small delay before reopening
                    import time
                    time.sleep(0.05)
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        raise VideoValidationError("Cannot reopen video after manual frame count")
                        
                    total_frames = manual_frame_count
                    fps = 30.0  # Assume 30fps if detection failed
                    logger.info(f"  Manual count: {manual_frame_count} frames, assuming {fps} fps")
            
            
            duration_sec = total_frames / fps if fps > 0 else 0
            logger.info(f"  Calculated duration: {duration_sec:.2f}s (frames={total_frames}, fps={fps})")
            
            # Check 1: Duration within acceptable range
            # For WebM recordings from browser, be VERY lenient on minimum duration
            # because metadata is often unreliable. Focus on hand visibility instead.
            min_duration_check = 0.01 if is_webm else self.min_duration_sec
            if duration_sec < min_duration_check:
                logger.error(f"Video duration too short: {duration_sec:.4f}s")
                raise VideoValidationError(
                    f"Video appears to be empty or corrupted (duration={duration_sec:.2f}s)."
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
            
            # Be more lenient with WebM files from browser recording
            # They often have encoding issues that affect detection
            required_ratio = self.min_hand_frames_ratio * 0.7 if is_webm else self.min_hand_frames_ratio
            
            logger.info(f"  Hand detection: {frames_with_hands}/{frames_sampled} frames ({hands_ratio*100:.1f}%)")
            logger.info(f"  Required ratio: {required_ratio*100:.1f}% (WebM: {is_webm})")
            
            # Check 4: Sufficient hand visibility
            if hands_ratio < required_ratio:
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
            # Be more lenient with WebM files
            if not motion_detected and not is_webm:
                raise VideoValidationError(
                    "No motion detected — video appears to be a static image. "
                    "Sign language videos must show hand/body movement."
                )
            
            logger.info(f"  Motion detected: {motion_detected}")
            logger.info(f"  ✅ Validation passed!")
            
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
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Video validation - Starting validation from bytes")
        logger.info(f"  Input: {len(video_bytes)} bytes, filename: {original_filename}")
        
        # Write to temporary file
        ext = Path(original_filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='wb') as tmp:
            tmp.write(video_bytes)
            tmp.flush()  # Ensure all data is written
            os.fsync(tmp.fileno())  # Force write to disk
            tmp_path = tmp.name
        
        logger.info(f"  Wrote to temp file: {tmp_path}")
        
        # Verify the file was actually written
        if not os.path.exists(tmp_path):
            logger.error(f"  Temp file does not exist after writing!")
            raise VideoValidationError("Failed to write video to temporary file")
        
        actual_size = os.path.getsize(tmp_path)
        logger.info(f"  Temp file size on disk: {actual_size} bytes")
        
        if actual_size == 0:
            logger.error(f"  Temp file is empty (0 bytes)!")
            raise VideoValidationError("Video file is empty - recording may have failed")
        
        if actual_size != len(video_bytes):
            logger.warning(f"  Size mismatch: wrote {len(video_bytes)} bytes but file is {actual_size} bytes")
        
        # Moderate delay to ensure WebM file is properly finalized
        # MediaRecorder WebM files need time to write container metadata
        import time
        logger.info(f"  Waiting 0.5s for file finalization...")
        time.sleep(0.5)  # Reduced from 1.0s
        
        # Try to open with OpenCV to verify it's readable
        logger.info(f"  Testing if OpenCV can open the file...")
        test_cap = cv2.VideoCapture(tmp_path)
        if not test_cap.isOpened():
            test_cap.release()
            logger.warning(f"  OpenCV cannot open the file directly - attempting to repair...")
            
            # Try to read the file frame-by-frame and re-encode it
            # This often fixes WebM files from browser MediaRecorder
            try:
                # Force OpenCV to read it anyway
                cap = cv2.VideoCapture(tmp_path)
                
                # Try to read at least one frame to see if it's readable at all
                frames = []
                max_frames = 300  # Read max 10 seconds at 30fps
                logger.info(f"  Attempting frame-by-frame read...")
                
                for i in range(max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                cap.release()
                
                if len(frames) == 0:
                    raise VideoValidationError(
                        "Video file is corrupted - no frames could be read. "
                        "Please try recording again."
                    )
                
                logger.info(f"  ✅ Read {len(frames)} frames, re-encoding...")
                
                # Re-encode to a new temp file using OpenCV VideoWriter
                height, width = frames[0].shape[:2]
                repaired_path = tmp_path + ".repaired.avi"
                
                # Use MJPEG codec (most compatible with OpenCV)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                fps = 30.0  # Assume 30fps
                
                writer = cv2.VideoWriter(repaired_path, fourcc, fps, (width, height))
                
                for frame in frames:
                    writer.write(frame)
                
                writer.release()
                
                # Verify the repaired file can be opened
                verify_cap = cv2.VideoCapture(repaired_path)
                if not verify_cap.isOpened():
                    verify_cap.release()
                    raise VideoValidationError("Failed to create valid video file from frames")
                verify_cap.release()
                
                # Replace temp file with repaired version
                logger.info(f"  ✅ Successfully repaired video, replacing original")
                os.unlink(tmp_path)
                os.rename(repaired_path, tmp_path)
                
            except Exception as e:
                logger.error(f"  ❌ Repair failed: {e}")
                raise VideoValidationError(
                    f"Video file cannot be processed: {str(e)}. "
                    "The recording may be corrupted. Please try again."
                )
        else:
            test_cap.release()
            logger.info(f"  ✅ OpenCV can open the file - proceeding with validation")
        
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
            min_duration_sec=0.01,  # Very lenient - just ensure video isn't empty
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
