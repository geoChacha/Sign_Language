"""
10_slp_render.py - Render sign language GIF from text sentence.

Run:
    python 10_slp_render.py --sentence "I need help" --out output.gif
    python 10_slp_render.py --word "hello" --out hello.gif
    python 10_slp_render.py --list-words          # show all available words

Requires trained model: checkpoints/slp_best_model.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

import config as cfg
from slp_model import SignProductionModel


# ── MediaPipe hand/pose connection graph ──────────────────────────────────────
# Hand connections: 21 landmarks, standard MediaPipe topology
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]

# Pose connections (33 landmarks) - upper body only for clarity
POSE_CONNECTIONS = [
    (11,12),             # shoulders
    (11,13),(13,15),     # left arm
    (12,14),(14,16),     # right arm
    (11,23),(12,24),     # torso sides
    (23,24),             # hips
]

# Pose landmark indices used (MediaPipe 33-point)
POSE_NOSE     = 0
POSE_L_EYE    = 2
POSE_R_EYE    = 5
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_ELBOW  = 13
POSE_R_ELBOW  = 14
POSE_L_WRIST  = 15
POSE_R_WRIST  = 16
POSE_L_HIP    = 23
POSE_R_HIP    = 24


def load_model(device):
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "slp_best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No trained model found at {ckpt_path}")
        print("        Run: python 9_slp_train.py first")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    mc   = ckpt["config"]
    model = SignProductionModel(**mc)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[Render] Loaded model from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f})")
    return model


def load_vocab():
    vocab_path = os.path.join(cfg.CHECKPOINTS_DIR, "slp_vocab.json")
    if not os.path.exists(vocab_path):
        print("[ERROR] slp_vocab.json not found - run 9_slp_train.py first")
        sys.exit(1)
    with open(vocab_path) as f:
        v = json.load(f)
    return v["idx_to_gloss"], v["gloss_to_idx"]


def pose_to_frame(pose_vec: np.ndarray, W: int = 420, H: int = 500,
                  word_label: str = "", frame_num: int = 0,
                  total_frames: int = 1) -> np.ndarray:
    """
    Render one frame of pose_vec (225,) as a clean skeleton image.

    Layout of pose_vec (225 dims):
      [0:63]   = left hand  (21 landmarks × 3 coords xyz)
      [63:126] = right hand (21 landmarks × 3 coords xyz)
      [126:225]= pose       (33 landmarks × 3 coords xyz, only first 99 used)
    """

    # ── Canvas ──────────────────────────────────────────────────────────────
    canvas = np.full((H, W, 3), (15, 17, 26), dtype=np.uint8)  # dark navy

    # Subtle grid
    for x in range(0, W, 60):
        cv2.line(canvas, (x, 0), (x, H), (22, 25, 38), 1)
    for y in range(0, H, 60):
        cv2.line(canvas, (0, y), (W, y), (22, 25, 38), 1)

    # ── Colour palette ───────────────────────────────────────────────────────
    C_POSE     = (80,  160, 255)   # blue  - body
    C_LHAND    = (80,  220, 160)   # green - left hand
    C_RHAND    = (255, 120,  60)   # orange - right hand
    C_JOINT    = (220, 230, 255)   # white-blue - joints
    C_FINGERTIP= (255, 255, 100)   # yellow - fingertips
    C_LABEL_BG = (25,  30,  50)
    C_LABEL    = (200, 215, 255)

    def to_px(x, y):
        """Normalised [0,1] coords → pixel coords, centered in canvas."""
        # MediaPipe gives x,y in [0,1] relative to image
        # We remap to fill 80% of canvas with margin
        margin_x = int(W * 0.10)
        margin_y = int(H * 0.08)
        px = int(margin_x + x * (W - 2 * margin_x))
        py = int(margin_y + y * (H - 2 * margin_y - 60))  # 60px for label
        return (px, py)

    # ── Parse keypoints ──────────────────────────────────────────────────────
    lh_pts = pose_vec[:63].reshape(21, 3)      # (21, 3) xyz
    rh_pts = pose_vec[63:126].reshape(21, 3)   # (21, 3) xyz
    p_pts  = pose_vec[126:].reshape(33, 3)     # (33, 3) xyz

    lh_visible  = lh_pts.sum() != 0
    rh_visible  = rh_pts.sum() != 0
    pose_visible = p_pts.sum() != 0

    def draw_hand(pts, color, fingertip_color):
        """Draw hand skeleton from 21 landmarks."""
        px_pts = [to_px(p[0], p[1]) for p in pts]
        # Connections
        for a, b in HAND_CONNECTIONS:
            pa, pb = px_pts[a], px_pts[b]
            # Skip if points are at origin (missing)
            if pts[a].sum() == 0 or pts[b].sum() == 0:
                continue
            cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
        # Joints
        fingertips = {4, 8, 12, 16, 20}
        for i, pt in enumerate(px_pts):
            if pts[i].sum() == 0:
                continue
            if i in fingertips:
                cv2.circle(canvas, pt, 5, fingertip_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 5, color, 1, cv2.LINE_AA)
            elif i == 0:
                cv2.circle(canvas, pt, 6, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, 3, C_JOINT, -1, cv2.LINE_AA)

    def draw_pose(pts):
        """Draw body skeleton from pose landmarks."""
        px_pts = [to_px(p[0], p[1]) for p in pts]
        # Connections
        for a, b in POSE_CONNECTIONS:
            if a >= len(pts) or b >= len(pts): continue
            if pts[a].sum() == 0 or pts[b].sum() == 0: continue
            cv2.line(canvas, px_pts[a], px_pts[b], C_POSE, 3, cv2.LINE_AA)
        # Key joints
        key_joints = [POSE_L_SHOULDER, POSE_R_SHOULDER,
                      POSE_L_ELBOW, POSE_R_ELBOW,
                      POSE_L_WRIST, POSE_R_WRIST,
                      POSE_L_HIP, POSE_R_HIP]
        for j in key_joints:
            if j < len(pts) and pts[j].sum() != 0:
                cv2.circle(canvas, px_pts[j], 5, C_JOINT, -1, cv2.LINE_AA)

        # Head: draw circle between eyes if available
        if pts[POSE_NOSE].sum() != 0:
            nose_px = px_pts[POSE_NOSE]
            cv2.circle(canvas, nose_px, 22, (30, 35, 55), -1)
            cv2.circle(canvas, nose_px, 22, C_POSE, 2, cv2.LINE_AA)
            # Eyes
            if pts[POSE_L_EYE].sum() != 0:
                cv2.circle(canvas, px_pts[POSE_L_EYE], 3, C_JOINT, -1)
            if pts[POSE_R_EYE].sum() != 0:
                cv2.circle(canvas, px_pts[POSE_R_EYE], 3, C_JOINT, -1)

    # Draw in order: pose first (behind), then hands on top
    if pose_visible:
        draw_pose(p_pts)
    if lh_visible:
        draw_hand(lh_pts, C_LHAND, (180, 255, 200))
    if rh_visible:
        draw_hand(rh_pts, C_RHAND, (255, 200, 100))

    # ── Legend dots ──────────────────────────────────────────────────────────
    cv2.circle(canvas, (15, 15), 5, C_LHAND, -1)
    cv2.putText(canvas, "L", (24, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_LHAND, 1)
    cv2.circle(canvas, (15, 30), 5, C_RHAND, -1)
    cv2.putText(canvas, "R", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_RHAND, 1)

    # ── Bottom label ─────────────────────────────────────────────────────────
    label_y = H - 55
    cv2.rectangle(canvas, (0, label_y), (W, H), C_LABEL_BG, -1)
    cv2.line(canvas, (0, label_y), (W, label_y), C_POSE, 1)

    if word_label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(1.2, 10.0 / max(len(word_label), 1))
        (tw, th), _ = cv2.getTextSize(word_label.upper(), font, scale, 2)
        tx = (W - tw) // 2
        cv2.putText(canvas, word_label.upper(), (tx, label_y + 35),
                    font, scale, C_LABEL, 2, cv2.LINE_AA)

    # Progress bar
    if total_frames > 1:
        prog = int((W - 40) * frame_num / (total_frames - 1))
        cv2.rectangle(canvas, (20, H-12), (W-20, H-5), (35, 40, 60), -1)
        cv2.rectangle(canvas, (20, H-12), (20+prog, H-5), C_POSE, -1)

    return canvas


def render_gif(pose_sequence: np.ndarray, word_labels: list,
               out_path: str, fps: int = 15):
    """
    pose_sequence : (T, 225) numpy array
    word_labels   : list of (start_frame, word) for label timing
    out_path      : output GIF path
    fps           : frames per second
    """
    T = pose_sequence.shape[0]
    duration_ms = int(1000 / fps)

    # Build label-per-frame lookup
    label_at_frame = {}
    for i, (start, word) in enumerate(word_labels):
        end = word_labels[i+1][0] if i+1 < len(word_labels) else T
        for f in range(start, end):
            label_at_frame[f] = word

    gif_frames = []
    for fi in range(T):
        label = label_at_frame.get(fi, "")
        frame = pose_to_frame(pose_sequence[fi], word_label=label,
                              frame_num=fi, total_frames=T)
        gif_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    # Hold last frame for 1 second
    for _ in range(fps):
        gif_frames.append(gif_frames[-1].copy())

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[Render] Saved {T}-frame GIF → {out_path}  ({size_kb:.0f} KB)")


def sentence_to_gloss_indices(sentence: str, gloss_to_idx: dict) -> list:
    """
    Simple word-level mapping. Lowercases and strips punctuation.
    Unknown words are skipped with a warning.
    """
    import re
    words = re.sub(r"[^a-zA-Z ]", "", sentence).lower().split()
    indices = []
    skipped = []
    for w in words:
        if w in gloss_to_idx:
            indices.append(gloss_to_idx[w])
        else:
            skipped.append(w)
    if skipped:
        print(f"[Render] WARNING: words not in vocabulary, skipped: {skipped}")
    return indices


def main():
    parser = argparse.ArgumentParser(description="Render ASL sign GIF from text")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sentence", type=str, help="Full sentence to sign")
    group.add_argument("--word",     type=str, help="Single word to sign")
    group.add_argument("--list-words", action="store_true",
                       help="List all words in vocabulary")
    parser.add_argument("--out",    type=str, default="sign_output.gif",
                        help="Output GIF path (default: sign_output.gif)")
    parser.add_argument("--fps",    type=int, default=15,
                        help="Output GIF frame rate (default: 15)")
    parser.add_argument("--frames-per-word", type=int, default=64,
                        help="Frames generated per word (default: 64)")
    parser.add_argument("--transition", type=int, default=8,
                        help="Transition frames between words (default: 8)")
    args = parser.parse_args()

    idx_to_gloss, gloss_to_idx = load_vocab()

    if args.list_words:
        print("\n[Vocabulary] Available words:")
        words = sorted(gloss_to_idx.keys())
        for i in range(0, len(words), 6):
            print("  " + "  ".join(f"{w:<18}" for w in words[i:i+6]))
        print(f"\nTotal: {len(words)} words")
        return

    device = torch.device(cfg.DEVICE)
    model  = load_model(device)

    if args.word:
        word = args.word.lower().strip()
        if word not in gloss_to_idx:
            print(f"[ERROR] '{word}' not in vocabulary. Run --list-words to see available words.")
            sys.exit(1)
        gloss_indices = [gloss_to_idx[word]]
        word_labels   = [(0, word)]
        sentence_text = word
    else:
        gloss_indices = sentence_to_gloss_indices(args.sentence, gloss_to_idx)
        if not gloss_indices:
            print("[ERROR] No known words found in sentence.")
            sys.exit(1)
        # Build word labels with timing
        word_labels = []
        frame_cursor = 0
        words_used = [w for w in args.sentence.lower().split()
                      if w in gloss_to_idx]
        for w in words_used:
            word_labels.append((frame_cursor, w))
            frame_cursor += args.frames_per_word + args.transition
        sentence_text = args.sentence

    print(f"[Render] Generating: '{sentence_text}'")
    print(f"[Render] Words: {[idx_to_gloss[str(i)] for i in gloss_indices]}")

    # Generate pose sequence
    gloss_tensor = torch.tensor(gloss_indices, dtype=torch.long)
    with torch.no_grad():
        pose_seq = model.generate_sentence(
            gloss_indices,
            n_frames_per_word=args.frames_per_word,
            transition_frames=args.transition,
        )   # (T_total, 225)

    pose_np = pose_seq.cpu().numpy()
    print(f"[Render] Generated {pose_np.shape[0]} frames")

    render_gif(pose_np, word_labels, args.out, fps=args.fps)


if __name__ == "__main__":
    main()