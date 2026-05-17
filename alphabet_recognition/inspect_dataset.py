from __future__ import annotations

import argparse
from pathlib import Path

from dataset import load_right_hand_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the PSL alphabet dataset.")
    parser.add_argument(
        "--db",
        default=str(Path("previous") / "main_dataset.db"),
        help="Path to main_dataset.db",
    )
    args = parser.parse_args()

    dataset = load_right_hand_dataset(args.db)
    print(f"Samples: {len(dataset.labels)}")
    print(f"Features: {dataset.features.shape[1]}")
    print(f"Classes: {len(dataset.class_counts)}")
    print()

    for label, count in sorted(
        dataset.class_counts.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{label}\t{count}")


if __name__ == "__main__":
    main()
