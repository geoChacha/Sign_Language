"""
Pipeline Orchestrator for ANN-Based PSL Alphabet Recognition System

Single entry point for all pipeline operations.

Usage:
    python run_pipeline.py --mode train
    python run_pipeline.py --mode evaluate
    python run_pipeline.py --mode demo [--threshold 0.75]
    python run_pipeline.py --mode all
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import CONFIDENCE_THRESHOLD, MODEL_PATH
from dataset_loader import load_dataset
from preprocessor import normalize_hand_coords
from model import AlphabetClassifier
from train import train_model
from evaluate import evaluate_model, load_model, save_confusion_matrix, save_evaluation_report
from demo import run_demo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_and_normalise():
    """Load dataset and normalise all splits. Returns splits + label_map."""
    logger.info("Loading dataset …")
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), label_map = load_dataset()

    X_tr = np.array([normalize_hand_coords(x) for x in X_tr])
    X_v  = np.array([normalize_hand_coords(x) for x in X_v])
    X_te = np.array([normalize_hand_coords(x) for x in X_te])

    logger.info(f"Dataset ready — {len(label_map)} classes | "
                f"train={len(X_tr)}  val={len(X_v)}  test={len(X_te)}")
    return (X_tr, y_tr), (X_v, y_v), (X_te, y_te), label_map


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_train():
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)

    train_data, val_data, test_data, label_map = _load_and_normalise()
    num_classes = len(label_map)

    model = AlphabetClassifier(num_classes=num_classes)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters, "
                f"{num_classes} output classes")

    train_model(model, train_data, val_data, test_data, label_map)
    print("✓ Training complete.")


def run_evaluate():
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    if not Path(MODEL_PATH).exists():
        print(f"Error: model not found at {MODEL_PATH}. Run --mode train first.")
        sys.exit(1)

    _, _, (X_te, y_te), _ = _load_and_normalise()

    model, label_map = load_model(MODEL_PATH)
    metrics = evaluate_model(model, (X_te, y_te), label_map)
    save_confusion_matrix(metrics['confusion_matrix'], metrics['label_names'])
    save_evaluation_report(metrics)
    print(f"✓ Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")


def run_demo_mode(threshold: float):
    print("\n" + "=" * 60)
    print("Starting Real-Time Demo")
    print("=" * 60)

    if not Path(MODEL_PATH).exists():
        print(f"Error: model not found at {MODEL_PATH}. Run --mode train first.")
        sys.exit(1)

    run_demo(threshold=threshold)


def run_all():
    print("\n" + "=" * 60)
    print("Running Full Pipeline")
    print("=" * 60)

    t0 = time.time()

    # 1. Train
    run_train()

    # 2. Evaluate
    run_evaluate()

    # 3. Demo
    run_demo_mode(CONFIDENCE_THRESHOLD)

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PSL Alphabet Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "demo", "all"],
        required=True,
        help="Pipeline mode",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for demo (default: {CONFIDENCE_THRESHOLD})",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        print(f"Error: --threshold must be in [0, 1], got {args.threshold}")
        sys.exit(1)

    dispatch = {
        "train":    run_train,
        "evaluate": run_evaluate,
        "demo":     lambda: run_demo_mode(args.threshold),
        "all":      run_all,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
