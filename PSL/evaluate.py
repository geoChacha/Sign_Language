"""
Evaluation Module for ANN-Based PSL Alphabet Recognition System

Computes test-set metrics, generates confusion matrices, and saves a
comprehensive evaluation report.

Metrics:
    - Overall accuracy
    - Per-class precision, recall, F1-score
    - Macro-averaged and weighted-averaged F1
    - Confusion matrix (text + PNG heatmap)
    - Problematic classes (F1 < 0.70)
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on Windows without display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from typing import Dict, Tuple, List
import logging

from config import (
    MODEL_PATH,
    EVAL_REPORT_PATH,
    CONFUSION_MATRIX_TXT,
    CONFUSION_MATRIX_PNG,
    BATCH_SIZE,
)
from model import AlphabetClassifier

logger = logging.getLogger(__name__)


def load_model(model_path: str = MODEL_PATH) -> Tuple[AlphabetClassifier, Dict[int, str]]:
    """
    Load a trained model and its label map from a checkpoint file.

    Args:
        model_path: Path to the .pt checkpoint saved by train.py

    Returns:
        (model, label_map)

    Raises:
        FileNotFoundError: if model_path does not exist
        RuntimeError:      if checkpoint format is unexpected
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run train.py first to train and save the model."
        )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Support both new dict-format and legacy state-dict-only format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint['num_classes']
        label_map   = checkpoint['label_map']
        state_dict  = checkpoint['model_state_dict']
    else:
        # Legacy: raw state dict — infer num_classes from output layer
        state_dict  = checkpoint
        num_classes = state_dict['fc3.weight'].shape[0]
        label_map   = {i: str(i) for i in range(num_classes)}
        logger.warning("Legacy checkpoint format detected — label names unavailable.")

    model = AlphabetClassifier(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded: {num_classes} classes from {model_path}")
    return model, label_map


def evaluate_model(
    model: AlphabetClassifier,
    test_data: Tuple[np.ndarray, np.ndarray],
    label_map: Dict[int, str],
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """
    Evaluate model on the test set and compute all metrics.

    Args:
        model:      Trained AlphabetClassifier (eval mode)
        test_data:  (X_test, y_test)
        label_map:  {int → alphabet string}
        batch_size: Batch size for inference

    Returns:
        Dict with keys: accuracy, confusion_matrix, classification_report,
                        top_confused_pairs, label_names, num_samples
    """
    from torch.utils.data import TensorDataset, DataLoader

    X_test, y_test = test_data
    dataset    = TensorDataset(torch.from_numpy(X_test).float(),
                               torch.from_numpy(y_test).long())
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch_X, _ in loader:
            all_preds.append(model(batch_X).argmax(1).numpy())

    predictions = np.concatenate(all_preds)
    accuracy    = (predictions == y_test).mean()

    label_names = [label_map[i] for i in range(len(label_map))]
    cm          = confusion_matrix(y_test, predictions)
    report      = classification_report(
        y_test, predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Top confused pairs (off-diagonal)
    confused = [
        (label_map[i], label_map[j], int(cm[i, j]))
        for i in range(len(cm))
        for j in range(len(cm))
        if i != j and cm[i, j] > 0
    ]
    confused.sort(key=lambda x: x[2], reverse=True)

    logger.info(f"Test accuracy: {accuracy:.4f}")
    return {
        'accuracy':              accuracy,
        'confusion_matrix':      cm,
        'classification_report': report,
        'top_confused_pairs':    confused[:5],
        'label_names':           label_names,
        'num_samples':           len(y_test),
    }


def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    txt_path: str = CONFUSION_MATRIX_TXT,
    png_path: str = CONFUSION_MATRIX_PNG,
) -> None:
    """Save confusion matrix as a text file and a matplotlib heatmap PNG."""
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Text file ────────────────────────────────────────────────────────────
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Confusion Matrix — PSL Alphabet Recognition\n")
        f.write("Row = True Label  |  Column = Predicted Label\n")
        f.write("=" * 80 + "\n\n")
        header = "        " + "".join(f"{l:>6s}" for l in labels)
        f.write(header + "\n")
        for i, label in enumerate(labels):
            row = f"{label:>6s}  " + "".join(f"{cm[i, j]:>6d}" for j in range(len(labels)))
            f.write(row + "\n")
    logger.info(f"Confusion matrix (text) → {txt_path}")

    # ── PNG heatmap ───────────────────────────────────────────────────────────
    n = len(labels)
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, label='Count')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Confusion Matrix — PSL Alphabet Recognition', fontsize=13, pad=12)

    # Annotate cells (skip zeros for readability)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, str(val),
                        ha='center', va='center', fontsize=6,
                        color='white' if val > thresh else 'black')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Confusion matrix (PNG)  → {png_path}")


def save_evaluation_report(metrics: Dict, report_path: str = EVAL_REPORT_PATH) -> None:
    """Save a comprehensive evaluation report to a text file."""
    accuracy    = metrics['accuracy']
    report      = metrics['classification_report']
    top_confused = metrics['top_confused_pairs']
    label_names = metrics['label_names']
    num_samples = metrics['num_samples']

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "PSL Alphabet Recognition — Evaluation Report",
        "=" * 70,
        f"Overall Accuracy : {accuracy:.4f}",
        f"Test samples     : {num_samples}",
        f"Number of classes: {len(label_names)}",
        "",
        f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 56,
    ]

    problematic = []
    for label in label_names:
        if label in report:
            p  = report[label]['precision']
            r  = report[label]['recall']
            f1 = report[label]['f1-score']
            s  = report[label]['support']
            lines.append(f"{label:<12} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {s:>10.0f}")
            if f1 < 0.70:
                problematic.append((label, f1))

    lines += [
        "",
        f"Macro-avg  F1 : {report['macro avg']['f1-score']:.4f}",
        f"Weighted-avg F1: {report['weighted avg']['f1-score']:.4f}",
        "",
    ]

    if problematic:
        lines.append("Problematic classes (F1 < 0.70):")
        for label, f1 in sorted(problematic, key=lambda x: x[1]):
            lines.append(f"  {label:<12}  F1 = {f1:.4f}")
        lines.append("")

    if top_confused:
        lines.append("Top confused pairs:")
        for i, (a, b, cnt) in enumerate(top_confused, 1):
            lines.append(f"  {i}. {a} → {b} : {cnt} times")
        lines.append("")

    if accuracy < 0.80:
        lines.append("⚠  Accuracy below 80% — consider collecting more data or tuning hyperparameters.")

    text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text)

    try:
        print("\n" + text)
    except UnicodeEncodeError:
        print("\n" + text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
                          .decode(sys.stdout.encoding or 'utf-8', errors='replace'))

    logger.info(f"Evaluation report → {report_path}")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    from dataset_loader import load_dataset
    from preprocessor import normalize_hand_coords

    logger.info("Loading dataset …")
    _, _, (X_te, y_te), _ = load_dataset()
    X_te = np.array([normalize_hand_coords(x) for x in X_te])

    model, label_map = load_model()

    metrics = evaluate_model(model, (X_te, y_te), label_map)
    save_confusion_matrix(metrics['confusion_matrix'], metrics['label_names'])
    save_evaluation_report(metrics)
