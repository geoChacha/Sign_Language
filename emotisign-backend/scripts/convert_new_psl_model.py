"""
Convert NEW PSL sklearn MLP model (JSON format) to PyTorch checkpoint.

This script reads the model.json from NEW PSL directory and converts it
to a PyTorch .pt file compatible with psl_live_service.py.

Usage:
    python scripts/convert_new_psl_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.psl_live_service import AlphabetClassifier


def load_sklearn_model_json(json_path: Path) -> dict:
    """Load the sklearn MLP model exported as JSON."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def convert_to_pytorch(model_data: dict) -> tuple:
    """
    Convert sklearn MLP weights to PyTorch AlphabetClassifier.
    
    Returns:
        (model, label_map) tuple
    """
    num_classes = len(model_data["labels"])
    input_dim = model_data["inputSize"]
    hidden_layers = model_data["hiddenLayers"]
    
    # Validate architecture matches
    assert input_dim == 42, f"Expected input_dim=42, got {input_dim}"
    assert hidden_layers == [128, 64], f"Expected [128, 64], got {hidden_layers}"
    
    # Create PyTorch model
    model = AlphabetClassifier(
        input_dim=input_dim,
        hidden_dim_1=hidden_layers[0],
        hidden_dim_2=hidden_layers[1],
        num_classes=num_classes,
        dropout=0.3,  # dropout is disabled at inference anyway
    )
    
    # Convert sklearn weights to PyTorch
    # sklearn stores weights as (input, output) but PyTorch uses (output, input)
    weights = model_data["weights"]
    biases = model_data["biases"]
    
    # Layer 1: input → hidden1
    w1 = np.array(weights[0], dtype=np.float32).T  # transpose
    b1 = np.array(biases[0], dtype=np.float32)
    model.fc1.weight.data = torch.from_numpy(w1)
    model.fc1.bias.data = torch.from_numpy(b1)
    
    # Layer 2: hidden1 → hidden2
    w2 = np.array(weights[1], dtype=np.float32).T
    b2 = np.array(biases[1], dtype=np.float32)
    model.fc2.weight.data = torch.from_numpy(w2)
    model.fc2.bias.data = torch.from_numpy(b2)
    
    # Layer 3: hidden2 → output
    w3 = np.array(weights[2], dtype=np.float32).T
    b3 = np.array(biases[2], dtype=np.float32)
    model.fc3.weight.data = torch.from_numpy(w3)
    model.fc3.bias.data = torch.from_numpy(b3)
    
    # BatchNorm layers — sklearn doesn't have these, so we initialize them
    # to identity transforms (mean=0, std=1) which is what they'll converge
    # to during eval mode anyway
    model.bn1.weight.data.fill_(1.0)
    model.bn1.bias.data.fill_(0.0)
    model.bn1.running_mean.fill_(0.0)
    model.bn1.running_var.fill_(1.0)
    
    model.bn2.weight.data.fill_(1.0)
    model.bn2.bias.data.fill_(0.0)
    model.bn2.running_mean.fill_(0.0)
    model.bn2.running_var.fill_(1.0)
    
    # Create label map (index → Urdu character)
    label_map = {i: label for i, label in enumerate(model_data["labels"])}
    
    return model, label_map


def save_pytorch_checkpoint(
    model: nn.Module,
    label_map: dict,
    output_path: Path,
    metadata: dict,
    scaler_mean: list,
    scaler_std: list,
) -> None:
    """Save PyTorch checkpoint in the format expected by psl_live_service.py."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": len(label_map),
        "label_map": label_map,
        "input_dim": 42,
        # StandardScaler parameters — required for correct inference
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "metadata": {
            "source": "NEW PSL sklearn MLP",
            "accuracy": metadata.get("accuracy", 0.0),
            "samples": metadata.get("samples", 0),
            "classes": metadata.get("classes", 0),
            "normalization": "wrist_subtract_landmark9_scale_then_standardize",
        },
    }
    
    torch.save(checkpoint, output_path)
    print(f"✓ Saved PyTorch checkpoint: {output_path}")
    print(f"  Classes: {len(label_map)}")
    print(f"  Accuracy: {metadata.get('accuracy', 0.0):.4f}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    new_psl_dir = project_root / "NEW PSL"
    model_json_path = new_psl_dir / "web_model" / "model.json"
    metrics_json_path = new_psl_dir / "web_model" / "metrics.json"
    
    output_dir = project_root / "emotisign-backend" / "app" / "ml" / "models" / "psl"
    output_path = output_dir / "alphabet_classifier.pt"
    label_map_path = output_dir / "label_map.json"
    
    # Validate input files exist
    if not model_json_path.exists():
        print(f"ERROR: {model_json_path} not found")
        print("Run train_export.py in NEW PSL directory first")
        sys.exit(1)
    
    if not metrics_json_path.exists():
        print(f"WARNING: {metrics_json_path} not found, proceeding without metadata")
        metadata = {}
    else:
        with open(metrics_json_path, encoding="utf-8") as f:
            metadata = json.load(f)
    
    # Load sklearn model
    print(f"Loading sklearn model from {model_json_path}")
    model_data = load_sklearn_model_json(model_json_path)
    
    # Convert to PyTorch
    print("Converting to PyTorch...")
    model, label_map = convert_to_pytorch(model_data)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint (include scaler parameters for correct inference)
    save_pytorch_checkpoint(
        model, label_map, output_path, metadata,
        scaler_mean=model_data["featureMean"],
        scaler_std=model_data["featureStd"],
    )
    
    # Save label map separately (for easy inspection)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved label map: {label_map_path}")
    
    # Verify the model loads correctly
    print("\nVerifying model loads correctly...")
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    verify_model = AlphabetClassifier(num_classes=checkpoint["num_classes"])
    verify_model.load_state_dict(checkpoint["model_state_dict"])
    verify_model.eval()
    
    # Test inference with dummy input
    dummy_input = torch.randn(1, 42)
    with torch.no_grad():
        output = verify_model(dummy_input)
    
    assert output.shape == (1, len(label_map)), f"Expected shape (1, {len(label_map)}), got {output.shape}"
    print(f"✓ Model verification passed (output shape: {output.shape})")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Model: {output_path}")
    print(f"Label map: {label_map_path}")
    print(f"Classes: {len(label_map)} (full Urdu alphabet)")
    print(f"Accuracy: {metadata.get('accuracy', 0.0):.4f}")
    print("\nThe PSL live service will automatically use the new model.")


if __name__ == "__main__":
    main()
