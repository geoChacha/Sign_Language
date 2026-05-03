"""
compare_splits.py - Compare dataset splits to inform training decisions.

Run:
    python compare_splits.py
"""

import json
import os
from collections import defaultdict
import config as cfg

split_files = {
    "nslt_100":  os.path.join(cfg.DATASET_ROOT, "nslt_100.json"),
    "nslt_300":  os.path.join(cfg.DATASET_ROOT, "nslt_300.json"),
    "nslt_1000": os.path.join(cfg.DATASET_ROOT, "nslt_1000.json"),
    "nslt_2000": os.path.join(cfg.DATASET_ROOT, "nslt_2000.json"),
}

# Load missing
missing = set()
if os.path.exists(cfg.MISSING_TXT):
    with open(cfg.MISSING_TXT) as f:
        for line in f:
            vid = line.strip().split("/")[-1].replace(".mp4", "")
            if vid:
                missing.add(vid)

# Load gloss map
with open(cfg.JSON_WLASL) as f:
    wlasl = json.load(f)
idx_to_gloss = {}
for entry in wlasl:
    for inst in entry.get("instances", []):
        vid_id = str(inst.get("video_id", ""))
        # will be filled per-split below

print("=" * 70)
print(f"{'Split':<12} {'Classes':>7} {'Train':>7} {'Val':>5} {'Test':>5} {'Total':>7} {'Missing':>8}")
print("=" * 70)

split_data = {}
for name, path in split_files.items():
    if not os.path.exists(path):
        print(f"{name:<12} -- file not found, skipping")
        continue

    with open(path) as f:
        d = json.load(f)

    train = val = test = miss = 0
    classes = set()
    for vid_id, info in d.items():
        classes.add(info["action"][0])
        subset = info["subset"]
        is_missing = vid_id in missing
        vid_path = os.path.join(cfg.VIDEOS_DIR, f"{vid_id}.mp4")
        file_missing = is_missing or not os.path.exists(vid_path)

        if subset == "train":
            if file_missing: miss += 1
            else: train += 1
        elif subset == "val":
            if not file_missing: val += 1
        elif subset == "test":
            if not file_missing: test += 1

    total = train + val + test
    print(f"{name:<12} {len(classes):>7} {train:>7} {val:>5} {test:>5} {total:>7} {miss:>8}")
    split_data[name] = {"d": d, "classes": classes}

print("=" * 70)

# Per-class training sample count comparison for nslt_100 vs nslt_300
if "nslt_100" in split_data and "nslt_300" in split_data:
    print("\nPer-class training samples (nslt_100 vs nslt_300) - first 30 classes:")
    print(f"{'Class':>5}  {'Word':<20}  {'nslt_100':>8}  {'nslt_300':>8}  {'Gain':>6}")
    print("-" * 55)

    # Build gloss map from nslt_100
    idx_to_gloss = {}
    with open(cfg.JSON_WLASL) as f:
        wlasl = json.load(f)
    d100 = split_data["nslt_100"]["d"]
    for entry in wlasl:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            vid_id = str(inst.get("video_id", ""))
            if vid_id in d100:
                idx_to_gloss[d100[vid_id]["action"][0]] = gloss

    def count_train(d):
        counts = defaultdict(int)
        for vid_id, info in d.items():
            if info["subset"] == "train":
                vid_path = os.path.join(cfg.VIDEOS_DIR, f"{vid_id}.mp4")
                if os.path.exists(vid_path) and vid_id not in missing:
                    counts[info["action"][0]] += 1
        return counts

    c100 = count_train(split_data["nslt_100"]["d"])
    c300 = count_train(split_data["nslt_300"]["d"])

    classes = sorted(split_data["nslt_100"]["classes"])[:30]
    for cls in classes:
        gloss = idx_to_gloss.get(cls, f"class_{cls}")
        n100 = c100.get(cls, 0)
        n300 = c300.get(cls, 0)
        gain = n300 - n100
        bar = "+" * min(gain, 20)
        print(f"{cls:>5}  {gloss:<20}  {n100:>8}  {n300:>8}  {gain:>+6}  {bar}")

    # Summary stats
    all_classes = sorted(split_data["nslt_100"]["classes"])
    gains = [c300.get(c, 0) - c100.get(c, 0) for c in all_classes]
    avg_100 = sum(c100.get(c, 0) for c in all_classes) / len(all_classes)
    avg_300 = sum(c300.get(c, 0) for c in all_classes) / len(all_classes)
    zero_100 = sum(1 for c in all_classes if c100.get(c, 0) == 0)
    zero_300 = sum(1 for c in all_classes if c300.get(c, 0) == 0)

    print(f"\nSummary:")
    print(f"  Avg train samples/class:  nslt_100={avg_100:.1f}  nslt_300={avg_300:.1f}")
    print(f"  Classes with 0 training:  nslt_100={zero_100}      nslt_300={zero_300}")
    print(f"  Avg gain per class: +{sum(gains)/len(gains):.1f} samples")