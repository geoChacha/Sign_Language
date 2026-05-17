from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset import load_right_hand_dataset


def _as_json_array(arr: np.ndarray) -> list:
    return np.asarray(arr, dtype=np.float32).tolist()


def export_model(
    path: Path,
    model: MLPClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    confidence_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "psl-alphabet-mlp-json-v1",
        "inputSize": int(scaler.mean_.shape[0]),
        "hiddenLayers": [int(size) for size in model.hidden_layer_sizes],
        "activation": model.activation,
        "outputActivation": "softmax",
        "confidenceThreshold": confidence_threshold,
        "labels": [str(label) for label in label_encoder.classes_],
        "featureMean": _as_json_array(scaler.mean_),
        "featureStd": _as_json_array(scaler.scale_),
        "weights": [_as_json_array(w) for w in model.coefs_],
        "biases": [_as_json_array(b) for b in model.intercepts_],
        "preprocessing": {
            "landmarks": 21,
            "coordinates": "x,y",
            "steps": [
                "subtract landmark 0 wrist from all points",
                "divide by distance from wrist to landmark 9",
                "flatten to 42 values",
                "standardize with featureMean and featureStd",
            ],
        },
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PSL alphabet classifier and export browser JSON weights."
    )
    parser.add_argument(
        "--db",
        default=str(Path("previous") / "main_dataset.db"),
        help="Path to main_dataset.db",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("alphabet_recognition") / "web_model"),
        help="Directory for model.json and metrics.json",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()

    dataset = load_right_hand_dataset(args.db)
    X = dataset.features

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset.labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.15,
        random_state=args.seed,
        verbose=True,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    accuracy = float(accuracy_score(y_val, preds))
    report = classification_report(
        y_val,
        preds,
        target_names=[str(label) for label in label_encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_val, preds).tolist()

    out_dir = Path(args.out_dir)
    export_model(
        out_dir / "model.json",
        model,
        scaler,
        label_encoder,
        args.confidence_threshold,
    )

    metrics = {
        "accuracy": accuracy,
        "samples": int(len(dataset.labels)),
        "features": int(X.shape[1]),
        "classes": len(label_encoder.classes_),
        "classCounts": dataset.class_counts,
        "classificationReport": report,
        "confusionMatrix": matrix,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Saved: {out_dir / 'model.json'}")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
