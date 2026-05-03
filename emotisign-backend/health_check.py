"""
health_check.py - Verify ML service setup and dependencies.

Run this script to verify:
- Model files exist
- GPU availability
- MediaPipe initialization
- PyTorch installation
"""

import os
import sys


def check_model_files():
    """Verify model files exist."""
    print("=" * 60)
    print("Checking Model Files...")
    print("=" * 60)
    
    required_files = [
        "app/ml/models/wlasl100/best_model.pth",
        "app/ml/models/wlasl100/vocab.json"
    ]
    optional_files = [
        "app/ml/models/wlasl100/temperature.json"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ Found: {file_path} ({size_mb:.2f} MB)")
        else:
            print(f"✗ Missing: {file_path}")
            all_good = False
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path} (optional)")
        else:
            print(f"  Missing: {file_path} (optional, will use default)")
    
    return all_good


def check_pytorch():
    """Check PyTorch installation and GPU."""
    print("\n" + "=" * 60)
    print("Checking PyTorch...")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU memory: {props.total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ No GPU available, will use CPU")
            print("  For better performance, install CUDA and GPU-enabled PyTorch")
            return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch==2.0.1 torchvision==0.15.2")
        return False


def check_mediapipe():
    """Verify MediaPipe can initialize."""
    print("\n" + "=" * 60)
    print("Checking MediaPipe...")
    print("=" * 60)
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
        
        # Try to initialize Holistic
        try:
            holistic = mp.solutions.holistic.Holistic()
            holistic.close()
            print("✓ MediaPipe Holistic initialized successfully")
            return True
        except AttributeError:
            print("⚠ MediaPipe version >= 0.10 detected")
            print("  Recommended version: mediapipe==0.9.3")
            print("  Install with: pip install mediapipe==0.9.3")
            return False
    except ImportError:
        print("✗ MediaPipe not installed")
        print("  Install with: pip install mediapipe==0.9.3")
        return False


def check_opencv():
    """Check OpenCV installation."""
    print("\n" + "=" * 60)
    print("Checking OpenCV...")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("✗ OpenCV not installed")
        print("  Install with: pip install opencv-python==4.8.1.78")
        return False


def check_numpy():
    """Check NumPy installation."""
    print("\n" + "=" * 60)
    print("Checking NumPy...")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        return True
    except ImportError:
        print("✗ NumPy not installed")
        print("  Install with: pip install numpy==1.24.3")
        return False


def main():
    """Run all health checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "EmotiSign ML Service Health Check" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    checks = {
        "Model Files": check_model_files(),
        "PyTorch": check_pytorch(),
        "MediaPipe": check_mediapipe(),
        "OpenCV": check_opencv(),
        "NumPy": check_numpy(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! ML service is ready.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nQuick fix:")
        print("  pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
