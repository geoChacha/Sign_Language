#!/usr/bin/env python3
"""
Simple test runner that doesn't require SQLAlchemy or other app dependencies.

This script runs only the ML-specific tests without importing the full app.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_model_architecture():
    """Test model architecture."""
    print("\n" + "=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    try:
        import torch
        from app.ml.wlasl_model import SignLanguageTransformer
        
        # Test 1: Initialization
        print("\n1. Testing model initialization...")
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        print("   ✓ Model initialized")
        
        # Test 2: Forward pass
        print("2. Testing forward pass...")
        model.eval()
        x = torch.randn(1, 64, 126)
        output = model(x)
        assert output.shape == (1, 100), f"Expected shape (1, 100), got {output.shape}"
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")
        
        # Test 3: Batch processing
        print("3. Testing batch processing...")
        x = torch.randn(4, 64, 126)
        output = model(x)
        assert output.shape == (4, 100), f"Expected shape (4, 100), got {output.shape}"
        print(f"   ✓ Batch processing successful, output shape: {output.shape}")
        
        # Test 4: Parameter count
        print("4. Testing parameter count...")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        assert 200_000 < total_params < 600_000, f"Parameter count {total_params} outside expected range"
        print("   ✓ Parameter count is within expected range")
        
        # Test 5: Dropout in eval mode
        print("5. Testing dropout behavior...")
        x = torch.randn(1, 64, 126)
        output1 = model(x)
        output2 = model(x)
        assert torch.allclose(output1, output2), "Outputs differ (dropout not disabled)"
        print("   ✓ Dropout is disabled in eval mode")
        
        print("\n✓ All model architecture tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Model architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_service():
    """Test ML service."""
    print("\n" + "=" * 60)
    print("Testing ML Service")
    print("=" * 60)
    
    try:
        import numpy as np
        from app.ml.wlasl_service import WLASLModelService
        
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        # Test 1: Sequence resampling - upsampling
        print("\n1. Testing sequence resampling (upsampling)...")
        keypoints = np.random.randn(30, 126).astype(np.float32)
        resampled = service.resample_sequence(keypoints, target_frames=64)
        assert resampled.shape == (64, 126), f"Expected shape (64, 126), got {resampled.shape}"
        print(f"   ✓ Upsampling: {keypoints.shape} → {resampled.shape}")
        
        # Test 2: Sequence resampling - downsampling
        print("2. Testing sequence resampling (downsampling)...")
        keypoints = np.random.randn(100, 126).astype(np.float32)
        resampled = service.resample_sequence(keypoints, target_frames=64)
        assert resampled.shape == (64, 126), f"Expected shape (64, 126), got {resampled.shape}"
        print(f"   ✓ Downsampling: {keypoints.shape} → {resampled.shape}")
        
        # Test 3: Sequence resampling - padding
        print("3. Testing sequence resampling (padding)...")
        keypoints = np.random.randn(20, 126).astype(np.float32)
        resampled = service.resample_sequence(keypoints, target_frames=64)
        assert resampled.shape == (64, 126), f"Expected shape (64, 126), got {resampled.shape}"
        print(f"   ✓ Padding: {keypoints.shape} → {resampled.shape}")
        
        # Test 4: TTA variants
        print("4. Testing TTA variant generation...")
        keypoints = np.random.randn(100, 126).astype(np.float32)
        variants = service.build_tta_variants(keypoints)
        assert len(variants) == 4, f"Expected 4 variants, got {len(variants)}"
        for i, variant in enumerate(variants):
            assert variant.shape == (64, 126), f"Variant {i} has wrong shape: {variant.shape}"
        print(f"   ✓ Generated {len(variants)} TTA variants")
        
        # Test 5: Statistics
        print("5. Testing statistics...")
        stats = service.get_stats()
        assert stats["total_predictions"] == 0
        assert stats["device"] == "cpu"
        print(f"   ✓ Statistics: {stats}")
        
        print("\n✓ All ML service tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ ML service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "WLASL-100 Unit Tests" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # Run tests
    results["Model Architecture"] = test_model_architecture()
    results["ML Service"] = test_ml_service()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
