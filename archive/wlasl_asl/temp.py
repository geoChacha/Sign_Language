"""
6_calibrate_temperature.py
==========================
Calibrates a temperature parameter T so that model confidence scores
are meaningful on real-world (out-of-distribution) input.

The problem without calibration:
    A model trained on WLASL studio videos outputs softmax probabilities
    that are overconfident. On phone-camera input it may say "99% sure"
    when it's actually wrong. Temperature scaling divides the logits by T
    before softmax, spreading the distribution. T > 1 makes it less
    confident; T < 1 makes it more confident.

    This is the single most impactful post-training fix for real-world
    deployment without collecting new data.

How it works:
    We find the T that minimises Negative Log-Likelihood on the
    validation set. This is fast (no retraining, just 1 scalar to tune).

Run AFTER training:
    python 6_calibrate_temperature.py

Output:
    checkpoints/temperature.json   -- loaded automatically by 5_inference.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.optimize import minimize_scalar

import config as cfg
from model import SignLanguageTransformer
from dataset import WLASLDataset


def load_model(ckpt_path: str, device: torch.device) -> SignLanguageTransformer:
    ckpt = torch.load(ckpt_path, map_location=device)
    mc = ckpt["config"]
    model = SignLanguageTransformer(
        feature_dim=mc["feature_dim"],
        num_classes=mc["num_classes"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=0.0,
        max_seq_len=mc["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def collect_logits(model, loader, device):
    """Collect all logits and labels from the validation set."""
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with autocast():
                logits = model(x)
            all_logits.append(logits.cpu().float())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def nll_with_temperature(temperature: float, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Negative log-likelihood after temperature scaling."""
    scaled = logits / temperature
    log_probs = torch.log_softmax(scaled, dim=-1)
    nll = nn.NLLLoss()(log_probs, labels)
    return nll.item()


def compute_ece(logits: torch.Tensor, labels: torch.Tensor,
                temperature: float, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) - lower is better.
    Measures how well confidence scores match actual accuracy.
    """
    scaled_probs = torch.softmax(logits / temperature, dim=-1)
    confidences, predictions = scaled_probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].float().mean().item()
        bin_conf = confidences[mask].mean().item()
        bin_size = mask.sum().item() / len(confidences)
        ece += bin_size * abs(bin_acc - bin_conf)
    return ece * 100.0  # as percentage


def main():
    device = torch.device(cfg.DEVICE)
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    print("[Calibrate] Loading model...")
    model = load_model(ckpt_path, device)

    print("[Calibrate] Collecting validation logits...")
    val_ds = WLASLDataset("val", augment=False)
    loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)
    logits, labels = collect_logits(model, loader, device)

    # Baseline metrics (T=1.0)
    baseline_nll = nll_with_temperature(1.0, logits, labels)
    baseline_ece = compute_ece(logits, labels, 1.0)
    baseline_acc = (logits.argmax(dim=-1) == labels).float().mean().item() * 100
    print(f"\n[Before calibration]  T=1.00  NLL={baseline_nll:.4f}  ECE={baseline_ece:.2f}%  Acc={baseline_acc:.1f}%")

    # Find optimal T using scalar minimisation
    print("[Calibrate] Optimising temperature...")
    result = minimize_scalar(
        lambda t: nll_with_temperature(t, logits, labels),
        bounds=(0.1, 10.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    optimal_t = float(result.x)

    # Metrics after calibration
    calibrated_nll = nll_with_temperature(optimal_t, logits, labels)
    calibrated_ece = compute_ece(logits, labels, optimal_t)
    print(f"[After  calibration]  T={optimal_t:.4f}  NLL={calibrated_nll:.4f}  ECE={calibrated_ece:.2f}%  Acc={baseline_acc:.1f}%")
    print(f"\n  NLL improvement: {baseline_nll - calibrated_nll:+.4f}")
    print(f"  ECE improvement: {baseline_ece - calibrated_ece:+.2f}%  (lower is better)")

    # Save temperature
    t_path = os.path.join(cfg.CHECKPOINTS_DIR, "temperature.json")
    with open(t_path, "w") as f:
        json.dump({
            "temperature": optimal_t,
            "baseline_nll": baseline_nll,
            "baseline_ece": baseline_ece,
            "calibrated_nll": calibrated_nll,
            "calibrated_ece": calibrated_ece,
            "val_accuracy": baseline_acc,
        }, f, indent=2)
    print(f"\n[Calibrate] Saved to {t_path}")
    print("[Calibrate] 5_inference.py will load this automatically.")

    # Interpretation
    if optimal_t > 1.5:
        print(f"\n[NOTE] T={optimal_t:.2f} > 1.5 means the model was significantly overconfident.")
        print("       Confidence scores in inference are now more realistic.")
    elif optimal_t < 0.8:
        print(f"\n[NOTE] T={optimal_t:.2f} < 0.8 means the model was actually underconfident.")


if __name__ == "__main__":
    main()