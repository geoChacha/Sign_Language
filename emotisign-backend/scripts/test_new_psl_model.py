"""
Quick test to verify the NEW PSL model (37 classes) loads and runs correctly.

Usage:
    python scripts/test_new_psl_model.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.psl_live_service import AlphabetClassifier


def test_model_loading():
    """Test that the model loads with correct number of classes."""
    model_path = Path("app/ml/models/psl/alphabet_classifier.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run convert_new_psl_model.py first")
        return False
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    num_classes = checkpoint["num_classes"]
    print(f"✓ Model has {num_classes} classes")
    
    if num_classes != 37:
        print(f"ERROR: Expected 37 classes, got {num_classes}")
        return False
    
    # Load model
    model = AlphabetClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✓ Model loaded successfully")
    
    # Test inference
    dummy_input = torch.randn(1, 42)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
    
    print(f"✓ Inference successful (output shape: {logits.shape})")
    print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  Probs sum: {probs.sum():.4f} (should be ~1.0)")
    
    # Check label map
    label_map = checkpoint.get("label_map", {})
    if len(label_map) != num_classes:
        print(f"WARNING: Label map has {len(label_map)} entries, expected {num_classes}")
    else:
        print(f"✓ Label map has {len(label_map)} entries")
    
    # Show first 5 and last 5 labels
    print("\nFirst 5 labels:")
    for i in range(min(5, len(label_map))):
        print(f"  {i}: {label_map.get(i, 'MISSING')}")
    
    print("\nLast 5 labels:")
    for i in range(max(0, len(label_map) - 5), len(label_map)):
        print(f"  {i}: {label_map.get(i, 'MISSING')}")
    
    return True


def test_coordinate_normalization():
    """
    Test that coordinate normalization matches the NEW PSL JS reference exactly.
    
    Reference: NEW PSL/web/psl_alphabet_classifier.js normalizeLandmarks()
    Pipeline:
      1. subtract wrist (landmark 0)
      2. divide by distance(wrist, landmark 9)
      3. flatten
      4. standardize with featureMean / featureStd
    """
    import json
    from pathlib import Path
    from app.ml.psl_live_service import PSLLiveService

    # Load scaler params from model.json to verify against
    model_json_path = Path("../NEW PSL/web_model/model.json")
    if not model_json_path.exists():
        print("\nSkipping normalization test (NEW PSL/web_model/model.json not found)")
        return True

    with open(model_json_path, encoding="utf-8") as f:
        model_data = json.load(f)

    scaler_mean = np.array(model_data["featureMean"], dtype=np.float32)
    scaler_std = np.array(model_data["featureStd"], dtype=np.float32)

    # Create service and inject scaler params directly
    service = PSLLiveService(model_path="dummy", label_map_path="dummy")
    service._scaler_mean = scaler_mean
    service._scaler_std = scaler_std

    # Build a synthetic hand with 21 landmarks at known positions
    np.random.seed(42)
    pts = np.random.rand(21, 2).astype(np.float32) * 200  # pixel coords

    # Compute expected output using the JS reference algorithm in Python
    wrist = pts[0]
    relative = pts - wrist
    middle_mcp = relative[9]
    scale = float(np.hypot(middle_mcp[0], middle_mcp[1]))
    safe_scale = scale if scale > 1e-6 else 1.0
    normalized = relative / safe_scale
    features = normalized.flatten()
    std_safe = np.where(scaler_std > 1e-9, scaler_std, 1.0)
    expected = ((features - scaler_mean) / std_safe).astype(np.float32)

    # Compute actual output from service
    actual = service.normalize_coordinates(pts.flatten())

    print("\n" + "=" * 60)
    print("Normalization Pipeline Test (NEW PSL)")
    print("=" * 60)
    print(f"Input shape: {pts.flatten().shape}")
    print(f"Output shape: {actual.shape}")
    print(f"Max absolute difference from reference: {np.abs(actual - expected).max():.2e}")

    if not np.allclose(actual, expected, atol=1e-5):
        print("✗ MISMATCH — normalization does not match JS reference")
        return False

    print("✓ Normalization matches JS reference exactly")
    return True


def main():
    print("=" * 60)
    print("NEW PSL Model Test")
    print("=" * 60)
    print()
    
    success = True
    
    # Test 1: Model loading
    if not test_model_loading():
        success = False
    
    print()
    
    # Test 2: Coordinate normalization
    if not test_coordinate_normalization():
        success = False
    
    print()
    print("=" * 60)
    if success:
        print("✓ All tests passed!")
        print("The NEW PSL model (37 classes) is ready to use.")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
