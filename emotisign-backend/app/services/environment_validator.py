"""
environment_validator.py - Pre-recording environment validation for ASL recognition.

Validates recording environment to ensure optimal conditions:
- Background clarity (plain background preferred)
- Body visibility (arms, hands, torso must be visible)
- Distance from camera (safe distance for proper framing)
"""

import cv2
import numpy as np
from typing import Dict, Optional
import logging

try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available for environment validation")


logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """
    Validates recording environment for ASL sign recognition.
    
    Checks:
    1. Background clarity - plain backgrounds work best
    2. Body visibility - arms, hands, and torso must be visible
    3. Distance - user should be at safe distance (not too close/far)
    """
    
    def __init__(self):
        """Initialize environment validator with MediaPipe Holistic."""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is required for environment validation. "
                "Install with: pip install mediapipe==0.9.3"
            )
        
        self.holistic = _mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,  # Faster for validation
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        logger.info("EnvironmentValidator initialized")
    
    def validate_background(self, frame: np.ndarray) -> Dict:
        """
        Analyze background clarity using edge detection.
        
        Plain backgrounds have fewer edges, cluttered backgrounds have many edges.
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            {
                "status": "clear" | "acceptable" | "cluttered",
                "score": float (0.0-1.0, lower is better),
                "message": str
            }
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density (ratio of edge pixels to total pixels)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Determine status based on edge density
        if edge_density < 0.1:
            status = "clear"
            message = "Background is clear - excellent!"
        elif edge_density < 0.2:
            status = "acceptable"
            message = "Background is acceptable"
        else:
            status = "cluttered"
            message = "Background is too cluttered - use a plain background"
        
        return {
            "status": status,
            "score": float(edge_density),
            "message": message
        }
    
    def validate_body_visibility(self, frame: np.ndarray) -> Dict:
        """
        Check if arms, hands, and torso are visible using MediaPipe pose.
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            {
                "status": "visible" | "partial" | "not_visible",
                "landmarks_detected": {
                    "left_hand": bool,
                    "right_hand": bool,
                    "arms": bool,
                    "torso": bool
                },
                "message": str
            }
        """
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            result = self.holistic.process(rgb)
        except Exception as e:
            logger.warning(f"MediaPipe processing failed: {e}")
            return {
                "status": "not_visible",
                "landmarks_detected": {
                    "left_hand": False,
                    "right_hand": False,
                    "arms": False,
                    "torso": False
                },
                "message": "Unable to detect body - ensure you are in frame"
            }
        
        # Check hand landmarks
        left_hand_visible = result.left_hand_landmarks is not None
        right_hand_visible = result.right_hand_landmarks is not None
        
        # Check pose landmarks for arms and torso
        arms_visible = False
        torso_visible = False
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            
            # Check arms (shoulders to wrists)
            # Landmarks: 11=left_shoulder, 13=left_elbow, 15=left_wrist
            #            12=right_shoulder, 14=right_elbow, 16=right_wrist
            left_arm_points = [11, 13, 15]
            right_arm_points = [12, 14, 16]
            
            left_arm_visible = all(
                landmarks[i].visibility > 0.5 for i in left_arm_points
            )
            right_arm_visible = all(
                landmarks[i].visibility > 0.5 for i in right_arm_points
            )
            arms_visible = left_arm_visible and right_arm_visible
            
            # Check torso (shoulders to hips)
            # Landmarks: 11=left_shoulder, 12=right_shoulder,
            #            23=left_hip, 24=right_hip
            torso_points = [11, 12, 23, 24]
            torso_visible = all(
                landmarks[i].visibility > 0.5 for i in torso_points
            )
        
        # Determine overall status
        landmarks_detected = {
            "left_hand": left_hand_visible,
            "right_hand": right_hand_visible,
            "arms": arms_visible,
            "torso": torso_visible
        }
        
        detected_count = sum(landmarks_detected.values())
        
        if detected_count == 4:
            status = "visible"
            message = "All body parts visible - perfect!"
        elif detected_count >= 2:
            status = "partial"
            missing = [k for k, v in landmarks_detected.items() if not v]
            message = f"Partially visible - ensure {', '.join(missing)} are in frame"
        else:
            status = "not_visible"
            message = "Body not visible - step back and ensure full upper body is in frame"
        
        return {
            "status": status,
            "landmarks_detected": landmarks_detected,
            "message": message
        }
    
    def validate_distance(self, frame: np.ndarray) -> Dict:
        """
        Estimate distance using shoulder width in pixels.
        
        Safe distance range: 150-400 pixels shoulder width
        Optimal range: 200-350 pixels
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            {
                "status": "optimal" | "acceptable" | "out_of_range",
                "shoulder_width_px": int,
                "message": str
            }
        """
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            result = self.holistic.process(rgb)
        except Exception as e:
            logger.warning(f"MediaPipe processing failed: {e}")
            return {
                "status": "out_of_range",
                "shoulder_width_px": 0,
                "message": "Unable to detect shoulders - ensure you are in frame"
            }
        
        if not result.pose_landmarks:
            return {
                "status": "out_of_range",
                "shoulder_width_px": 0,
                "message": "Unable to detect shoulders - ensure you are in frame"
            }
        
        landmarks = result.pose_landmarks.landmark
        
        # Get shoulder landmarks (11=left_shoulder, 12=right_shoulder)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check visibility
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return {
                "status": "out_of_range",
                "shoulder_width_px": 0,
                "message": "Shoulders not clearly visible"
            }
        
        # Calculate shoulder width in pixels
        # MediaPipe returns normalized coordinates (0-1)
        frame_width = frame.shape[1]
        shoulder_width_px = int(
            abs(right_shoulder.x - left_shoulder.x) * frame_width
        )
        
        # Determine status based on shoulder width
        if 200 <= shoulder_width_px <= 350:
            status = "optimal"
            message = "Distance is optimal - perfect!"
        elif 150 <= shoulder_width_px < 200 or 350 < shoulder_width_px <= 400:
            status = "acceptable"
            if shoulder_width_px < 200:
                message = "Move slightly closer to camera"
            else:
                message = "Move slightly back from camera"
        else:
            status = "out_of_range"
            if shoulder_width_px < 150:
                message = "Too far from camera - move closer"
            else:
                message = "Too close to camera - move back"
        
        return {
            "status": status,
            "shoulder_width_px": shoulder_width_px,
            "message": message
        }
    
    async def validate_frame(self, frame: np.ndarray) -> Dict:
        """
        Comprehensive validation of a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
        Returns:
            {
                "background": {...},
                "body": {...},
                "distance": {...},
                "overall": "ready" | "not_ready"
            }
        """
        # Run all validations
        background = self.validate_background(frame)
        body = self.validate_body_visibility(frame)
        distance = self.validate_distance(frame)
        
        # Determine overall status
        # Ready if:
        # - Background is clear or acceptable
        # - Body is visible
        # - Distance is optimal or acceptable
        background_ok = background["status"] in ["clear", "acceptable"]
        body_ok = body["status"] == "visible"
        distance_ok = distance["status"] in ["optimal", "acceptable"]
        
        overall = "ready" if (background_ok and body_ok and distance_ok) else "not_ready"
        
        return {
            "background": background,
            "body": body,
            "distance": distance,
            "overall": overall
        }
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()
            logger.info("EnvironmentValidator closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# Singleton instance
_validator: Optional[EnvironmentValidator] = None


def get_validator() -> EnvironmentValidator:
    """Get or create environment validator singleton."""
    global _validator
    
    if _validator is None:
        _validator = EnvironmentValidator()
    
    return _validator


if __name__ == "__main__":
    # Test environment validator
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        # Create a test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("Testing EnvironmentValidator...")
        with EnvironmentValidator() as validator:
            result = await validator.validate_frame(test_frame)
            
            print("\nValidation Results:")
            print(f"Background: {result['background']['status']} - {result['background']['message']}")
            print(f"Body: {result['body']['status']} - {result['body']['message']}")
            print(f"Distance: {result['distance']['status']} - {result['distance']['message']}")
            print(f"Overall: {result['overall']}")
        
        print("\n✓ EnvironmentValidator test completed")
    
    asyncio.run(test())
