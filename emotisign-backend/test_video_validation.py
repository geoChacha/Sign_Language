#!/usr/bin/env python3
"""
Test script for video validation.

Usage:
    python test_video_validation.py path/to/video.mp4
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.video_validator import validate_sign_video, VideoValidationError


def test_video(video_path: str):
    """Test a single video file."""
    print(f"\n{'='*60}")
    print(f"Testing: {video_path}")
    print(f"{'='*60}\n")
    
    try:
        result = validate_sign_video(video_path)
        
        print("✅ VALIDATION PASSED\n")
        print("Metrics:")
        for key, value in result.items():
            print(f"  {key:25s} : {value}")
        
        print("\n📊 Analysis:")
        hand_ratio = result['hand_visibility_ratio']
        if hand_ratio >= 0.50:
            print(f"  • Excellent hand visibility ({hand_ratio*100:.0f}%)")
        elif hand_ratio >= 0.30:
            print(f"  • Acceptable hand visibility ({hand_ratio*100:.0f}%)")
        else:
            print(f"  • Low hand visibility ({hand_ratio*100:.0f}%) — might be rejected")
        
        if result['motion_detected']:
            print("  • Motion detected — video is not static")
        else:
            print("  • ⚠️ No motion detected — might be a static image")
        
        print(f"\n✅ This video will be ACCEPTED for processing")
        return True
        
    except VideoValidationError as e:
        print("❌ VALIDATION FAILED\n")
        print(f"Error: {e}\n")
        print("This video will be REJECTED.")
        return False
    
    except Exception as e:
        print("❌ UNEXPECTED ERROR\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_video_validation.py <video_path>")
        print("\nExample:")
        print("  python test_video_validation.py test_videos/sign_hello.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Error: File not found: {video_path}")
        sys.exit(1)
    
    success = test_video(video_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
