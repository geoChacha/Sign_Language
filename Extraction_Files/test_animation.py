"""
Quick animation test - View keypoints in motion
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

def test_animation(sign_name='basketball'):
    """Test animation for a specific sign"""
    
    # Load best keypoints
    with open('keypoints_full/extraction_results.json') as f:
        results = json.load(f)
    
    if sign_name not in results:
        print(f"Sign '{sign_name}' not found. Available: {list(results.keys())}")
        return
    
    # Get best instance
    instances = results[sign_name]
    best = max(instances, key=lambda x: x['quality']['metrics']['quality_score'])
    
    # Load keypoints
    keypoints = np.load(best['file'])
    
    print(f"Testing: {sign_name}")
    print(f"Video: {best['video_id']}")
    print(f"Frames: {len(keypoints)}")
    print(f"Quality: {best['quality']['metrics']['quality_score']:.3f}")
    print(f"\nClose the window to exit...")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame_idx):
        ax.clear()
        frame = keypoints[frame_idx]
        
        # Body landmarks
        body = frame[:33]
        
        # Key body points
        nose = body[0]
        left_shoulder = body[11]
        right_shoulder = body[12]
        left_elbow = body[13]
        right_elbow = body[14]
        left_wrist = body[15]
        right_wrist = body[16]
        left_hip = body[23]
        right_hip = body[24]
        
        # Plot body
        ax.scatter(body[:, 0], body[:, 1], c='blue', s=100, alpha=0.6, label='Body')
        
        # Draw skeleton
        # Arms
        if np.any(left_shoulder != 0) and np.any(left_elbow != 0):
            ax.plot([left_shoulder[0], left_elbow[0]], [left_shoulder[1], left_elbow[1]], 
                   'b-', linewidth=3, alpha=0.5)
        if np.any(left_elbow != 0) and np.any(left_wrist != 0):
            ax.plot([left_elbow[0], left_wrist[0]], [left_elbow[1], left_wrist[1]], 
                   'b-', linewidth=3, alpha=0.5)
        
        if np.any(right_shoulder != 0) and np.any(right_elbow != 0):
            ax.plot([right_shoulder[0], right_elbow[0]], [right_shoulder[1], right_elbow[1]], 
                   'b-', linewidth=3, alpha=0.5)
        if np.any(right_elbow != 0) and np.any(right_wrist != 0):
            ax.plot([right_elbow[0], right_wrist[0]], [right_elbow[1], right_wrist[1]], 
                   'b-', linewidth=3, alpha=0.5)
        
        # Torso
        if np.any(left_shoulder != 0) and np.any(right_shoulder != 0):
            ax.plot([left_shoulder[0], right_shoulder[0]], [left_shoulder[1], right_shoulder[1]], 
                   'b-', linewidth=4, alpha=0.5)
        if np.any(left_shoulder != 0) and np.any(left_hip != 0):
            ax.plot([left_shoulder[0], left_hip[0]], [left_shoulder[1], left_hip[1]], 
                   'b-', linewidth=3, alpha=0.5)
        if np.any(right_shoulder != 0) and np.any(right_hip != 0):
            ax.plot([right_shoulder[0], right_hip[0]], [right_shoulder[1], right_hip[1]], 
                   'b-', linewidth=3, alpha=0.5)
        if np.any(left_hip != 0) and np.any(right_hip != 0):
            ax.plot([left_hip[0], right_hip[0]], [left_hip[1], right_hip[1]], 
                   'b-', linewidth=4, alpha=0.5)
        
        # Hands
        left_hand = frame[33:54]
        right_hand = frame[54:75]
        
        if np.any(left_hand != 0):
            ax.scatter(left_hand[:, 0], left_hand[:, 1], c='red', s=50, alpha=0.7, label='Left Hand')
            # Connect wrist to hand
            if np.any(left_wrist != 0) and np.any(left_hand[0] != 0):
                ax.plot([left_wrist[0], left_hand[0, 0]], [left_wrist[1], left_hand[0, 1]], 
                       'r--', linewidth=2, alpha=0.3)
        
        if np.any(right_hand != 0):
            ax.scatter(right_hand[:, 0], right_hand[:, 1], c='green', s=50, alpha=0.7, label='Right Hand')
            # Connect wrist to hand
            if np.any(right_wrist != 0) and np.any(right_hand[0] != 0):
                ax.plot([right_wrist[0], right_hand[0, 0]], [right_wrist[1], right_hand[0, 1]], 
                       'g--', linewidth=2, alpha=0.3)
        
        # Highlight key points
        if np.any(nose != 0):
            ax.scatter(nose[0], nose[1], c='yellow', s=200, marker='o', edgecolors='black', linewidths=2, label='Head', zorder=10)
        if np.any(left_wrist != 0):
            ax.scatter(left_wrist[0], left_wrist[1], c='red', s=150, marker='s', edgecolors='black', linewidths=2, zorder=10)
        if np.any(right_wrist != 0):
            ax.scatter(right_wrist[0], right_wrist[1], c='green', s=150, marker='s', edgecolors='black', linewidths=2, zorder=10)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f'{sign_name.upper()} - Frame {frame_idx+1}/{len(keypoints)}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
    
    anim = FuncAnimation(fig, update, frames=len(keypoints), interval=33, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import sys
    
    # Get sign name from command line or use default
    sign_name = sys.argv[1] if len(sys.argv) > 1 else 'basketball'
    
    print("="*70)
    print("KEYPOINT ANIMATION TEST")
    print("="*70)
    print(f"\nTesting sign: {sign_name}")
    print("\nAvailable signs: basketball, city, orange, shirt, who")
    print("\nUsage: python test_animation.py <sign_name>")
    print("="*70 + "\n")
    
    test_animation(sign_name)
