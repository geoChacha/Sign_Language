"""
Quick animation test - View keypoints in motion (FULLY FIXED & IMPROVED)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# MediaPipe POSE_CONNECTIONS
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,31),(23,26),(26,28),(28,32),(24,26)
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(5,9),(9,10),(10,11),(11,12),
    (0,13),(9,13),(13,14),(14,15),(15,16),
    (0,17),(13,17),(17,18),(18,19),(19,20),
]

def is_valid(point):
    if not hasattr(point, '__array__'):
        return False
    point = np.asarray(point).ravel()
    if point.size < 2:
        return False
    return np.any(point[:2] != 0)

def draw_skeleton(ax, points, connections, color='blue', linewidth=3, alpha=0.7, linestyle='-'):
    n_points = len(points)
    for start_idx, end_idx in connections:
        if start_idx < n_points and end_idx < n_points:
            start_pt = points[start_idx]
            end_pt = points[end_idx]
            if is_valid(start_pt) and is_valid(end_pt):
                start_xy = np.asarray(start_pt)[:2]
                end_xy = np.asarray(end_pt)[:2]
                ax.plot([start_xy[0], end_xy[0]], [start_xy[1], end_xy[1]],
                       color=color, linewidth=linewidth, alpha=alpha,
                       linestyle=linestyle, zorder=5)

def get_available_signs(folder='keypoints_best'):
    return [f.replace('.npy', '') for f in os.listdir(folder) if f.endswith('.npy')]

def test_animation(sign_name='basketball', folder='keypoints_best'):
    npy_path = os.path.join(folder, f'{sign_name}.npy')

    if not os.path.exists(npy_path):
        available = get_available_signs(folder)
        print(f"Sign '{sign_name}' not found in '{folder}/'.")
        print(f"Available ({len(available)}): {', '.join(sorted(available))}")
        return

    keypoints = np.load(npy_path, allow_pickle=True)

    print(f"Sign    : {sign_name}")
    print(f"File    : {npy_path}")
    print(f"Frames  : {len(keypoints)}")
    print(f"Shape   : {keypoints.shape}")
    print(f"\nClose the window to exit...")

    fig, ax = plt.subplots(figsize=(14, 12))
    trail_length = 5

    def update(frame_idx):
        ax.clear()
        frame = keypoints[frame_idx]

        body = frame[:33]
        valid_body_mask = np.array([is_valid(body[i]) for i in range(33)])
        valid_body_pts = np.array([np.asarray(body[i])[:2] for i in range(33) if valid_body_mask[i]])

        if len(valid_body_pts) > 0:
            ax.scatter(valid_body_pts[:, 0], valid_body_pts[:, 1], c='blue', s=80, alpha=0.6, zorder=10, label='Body')

        draw_skeleton(ax, body, POSE_CONNECTIONS[:12], color='cyan', linewidth=5, alpha=0.85)
        draw_skeleton(ax, body, POSE_CONNECTIONS[12:], color='magenta', linewidth=4, alpha=0.8)

        left_hand = frame[33:54]
        left_valid_mask = np.array([is_valid(left_hand[i]) for i in range(21)])
        valid_lh_pts = np.array([np.asarray(left_hand[i])[:2] for i in range(21) if left_valid_mask[i]])

        if len(valid_lh_pts) > 0:
            ax.scatter(valid_lh_pts[:, 0], valid_lh_pts[:, 1], c='red', s=40, alpha=0.85, zorder=15, label='Left Hand')
            if np.sum(left_valid_mask) >= 5:
                draw_skeleton(ax, left_hand, HAND_CONNECTIONS, color='red', linewidth=2.5, alpha=0.95)

        right_hand = frame[54:75]
        right_valid_mask = np.array([is_valid(right_hand[i]) for i in range(21)])
        valid_rh_pts = np.array([np.asarray(right_hand[i])[:2] for i in range(21) if right_valid_mask[i]])

        if len(valid_rh_pts) > 0:
            ax.scatter(valid_rh_pts[:, 0], valid_rh_pts[:, 1], c='lime', s=40, alpha=0.85, zorder=15, label='Right Hand')
            if np.sum(right_valid_mask) >= 5:
                draw_skeleton(ax, right_hand, HAND_CONNECTIONS, color='lime', linewidth=2.5, alpha=0.95)

        for t in range(1, min(trail_length, frame_idx + 1)):
            prev_idx = frame_idx - t
            if prev_idx >= 0:
                prev_frame = keypoints[prev_idx]
                prev_body = prev_frame[:33]
                prev_valid_mask = np.array([is_valid(prev_body[i]) for i in range(33)])
                prev_valid_pts = np.array([np.asarray(prev_body[i])[:2] for i in range(33) if prev_valid_mask[i]])
                if len(prev_valid_pts) > 0:
                    ax.scatter(prev_valid_pts[:, 0], prev_valid_pts[:, 1],
                              c='gray', s=25, alpha=0.25/t, zorder=1)

        key_configs = [
            (0,  'gold',    'o', 220, 'Nose'),
            (15, 'red',     's', 180, 'L-Wrist'),
            (16, 'lime',    's', 180, 'R-Wrist'),
            (23, 'magenta', '^', 160, 'L-Hip'),
            (24, 'orange',  '^', 160, 'R-Hip')
        ]

        for idx, color, marker, size, label in key_configs:
            if idx < len(body) and is_valid(body[idx]):
                pt = np.asarray(body[idx])[:2]
                ax.scatter(pt[0], pt[1], c=color, s=size, marker=marker,
                          edgecolors='black', linewidths=3, zorder=25, label=label)

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, 1.45)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f'{sign_name.upper()} - Frame {frame_idx+1}/{len(keypoints)}',
                    fontsize=18, fontweight='bold', pad=25)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle=':', zorder=0)
        ax.set_facecolor('#f0f2f5')
        ax.set_xlabel('X (Normalized)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y (Normalized)', fontsize=13, fontweight='bold')

    anim = FuncAnimation(fig, update, frames=len(keypoints), interval=50, repeat=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    available = get_available_signs()

    print("=" * 70)
    print("KEYPOINT ANIMATION — keypoints_best/")
    print("=" * 70)
    print(f"Available ({len(available)}): {', '.join(sorted(available))}")
    print("Usage: python animate.py <sign_name>")
    print("=" * 70 + "\n")

    sign_name = sys.argv[1] if len(sys.argv) > 1 else 'basketball'
    test_animation(sign_name)