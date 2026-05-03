#!/usr/bin/env python3
"""
Simple test script to verify the WLASL-100 system works end-to-end.

This script tests:
1. Model loading
2. Keypoint extraction
3. Inference with dummy data
4. Environment validation
5. API endpoints (if server is running)

Usage:
    python test_system.py
"""

import os
import sys
import asyncio
import numpy as np
import cv2
import torch
from pathlib import Path

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import, but handle missing dependencies gracefully
try:
    from app.ml.wlasl_service import WLASLModelService
    from app.ml.keypoint_extractor import KeypointExtractor
    from app.services.environment_validator import EnvironmentValidator
    IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("Some tests may be skipped.")
    IMPORTS_OK = False


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"✗ {text}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


async def test_model_loading():
    """Test 1: Model Loading"""
    print_header("Test 1: Model Loading")
    
    if not IMPORTS_OK:
        print_error("Skipping test - imports failed")
        return False
    
    try:
        # Check if model files exist
        model_path = "app/ml/models/wlasl100/best_model.pth"
        vocab_path = "app/ml/models/wlasl100/vocab.json"
        
        if not os.path.exists(model_path):
            print_error(f"Model file not found: {model_path}")
            return False
        
        if not os.path.exists(vocab_path):
            print_error(f"Vocabulary file not found: {vocab_path}")
            return False
        
        print_success("Model files found")
        
        # Initialize service
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_info(f"Using device: {device}")
        
        service = WLASLModelService(
            model_path=model_path,
            vocab_path=vocab_path,
            device=device,
            use_tta=False  # Disable TTA for faster testing
        )
        
        # Load model
        await service.load_model()
        print_success("Model loaded successfully")
        
        # Check model is in eval mode
        if service.model.training:
            print_error("Model is in training mode (should be eval)")
            return False
        
        print_success("Model is in evaluation mode")
        
        # Check vocabulary loaded
        if len(service.vocab) != 100:
            print_error(f"Vocabulary has {len(service.vocab)} words (expected 100)")
            return False
        
        print_success(f"Vocabulary loaded: {len(service.vocab)} words")
        
        return True
        
    except Exception as e:
        print_error(f"Model loading failed: {e}")
        return False


async def test_keypoint_extraction():
    """Test 2: Keypoint Extraction"""
    print_header("Test 2: Keypoint Extraction")
    
    try:
        # Initialize extractor
        extractor = KeypointExtractor(model_complexity=2)
        print_success("KeypointExtractor initialized")
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract keypoints
        keypoints = extractor.extract_keypoints_from_frame(frame)
        
        # Check shape
        if keypoints.shape != (126,):
            print_error(f"Keypoints shape is {keypoints.shape} (expected (126,))")
            return False
        
        print_success(f"Keypoints extracted: shape {keypoints.shape}")
        
        # Check data type
        if keypoints.dtype != np.float32:
            print_error(f"Keypoints dtype is {keypoints.dtype} (expected float32)")
            return False
        
        print_success("Keypoints dtype is correct")
        
        # Cleanup
        extractor.close()
        
        return True
        
    except Exception as e:
        print_error(f"Keypoint extraction failed: {e}")
        return False


async def test_inference():
    """Test 3: Inference with Dummy Data"""
    print_header("Test 3: Inference with Dummy Data")
    
    try:
        # Initialize service
        device = "cuda" if torch.cuda.is_available() else "cpu"
        service = WLASLModelService(
            model_path="app/ml/models/wlasl100/best_model.pth",
            vocab_path="app/ml/models/wlasl100/vocab.json",
            device=device,
            use_tta=False
        )
        
        await service.load_model()
        
        # Create dummy keypoints (100 frames, 126 features)
        dummy_keypoints = np.random.randn(100, 126).astype(np.float32)
        print_info(f"Created dummy keypoints: shape {dummy_keypoints.shape}")
        
        # Run inference
        result = await service.predict(dummy_keypoints, use_tta=False)
        
        # Check result format
        required_keys = [
            "recognized_text",
            "glosses",
            "confidence",
            "top5_predictions",
            "frame_count",
            "processing_time_ms",
            "sign_language"
        ]
        
        for key in required_keys:
            if key not in result:
                print_error(f"Missing key in result: {key}")
                return False
        
        print_success("Result has all required keys")
        
        # Check values
        print_info(f"Recognized text: {result['recognized_text']}")
        print_info(f"Confidence: {result['confidence']:.3f}")
        print_info(f"Top 5: {result['glosses']}")
        print_info(f"Processing time: {result['processing_time_ms']}ms")
        
        # Check confidence is in valid range
        if not (0.0 <= result['confidence'] <= 1.0):
            print_error(f"Confidence {result['confidence']} is out of range [0, 1]")
            return False
        
        print_success("Confidence is in valid range")
        
        # Check top 5 predictions
        if len(result['top5_predictions']) != 5:
            print_error(f"Expected 5 predictions, got {len(result['top5_predictions'])}")
            return False
        
        print_success("Top 5 predictions returned")
        
        return True
        
    except Exception as e:
        print_error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_environment_validation():
    """Test 4: Environment Validation"""
    print_header("Test 4: Environment Validation")
    
    try:
        # Initialize validator
        validator = EnvironmentValidator()
        print_success("EnvironmentValidator initialized")
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Validate frame
        result = await validator.validate_frame(frame)
        
        # Check result format
        required_keys = ["background", "body", "distance", "overall"]
        
        for key in required_keys:
            if key not in result:
                print_error(f"Missing key in result: {key}")
                return False
        
        print_success("Result has all required keys")
        
        # Check values
        print_info(f"Background: {result['background']['status']}")
        print_info(f"Body: {result['body']['status']}")
        print_info(f"Distance: {result['distance']['status']}")
        print_info(f"Overall: {result['overall']}")
        
        # Check overall status
        if result['overall'] not in ['ready', 'not_ready']:
            print_error(f"Invalid overall status: {result['overall']}")
            return False
        
        print_success("Overall status is valid")
        
        return True
        
    except Exception as e:
        print_error(f"Environment validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_endpoints():
    """Test 5: API Endpoints (if server is running)"""
    print_header("Test 5: API Endpoints")
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/api/ml/health", timeout=2)
            if response.status_code == 200:
                print_success("Health endpoint is accessible")
                data = response.json()
                print_info(f"Status: {data.get('status')}")
                print_info(f"Model loaded: {data.get('model_loaded')}")
                print_info(f"Device: {data.get('device')}")
            else:
                print_error(f"Health endpoint returned {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print_error("Server is not running")
            print_info("Start server with: uvicorn main:app --reload")
            return False
        
        # Test stats endpoint
        response = requests.get(f"{base_url}/api/ml/stats", timeout=2)
        if response.status_code == 200:
            print_success("Stats endpoint is accessible")
            data = response.json()
            print_info(f"Total predictions: {data.get('total_predictions')}")
        else:
            print_error(f"Stats endpoint returned {response.status_code}")
            return False
        
        return True
        
    except ImportError:
        print_error("requests library not installed")
        print_info("Install with: pip install requests")
        return False
    except Exception as e:
        print_error(f"API endpoint test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "WLASL-100 System Verification" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # Run tests
    results["Model Loading"] = await test_model_loading()
    results["Keypoint Extraction"] = await test_keypoint_extraction()
    results["Inference"] = await test_inference()
    results["Environment Validation"] = await test_environment_validation()
    results["API Endpoints"] = await test_api_endpoints()
    
    # Print summary
    print_header("Summary")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! System is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
