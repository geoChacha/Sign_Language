"""
Fresh MediaPipe Keypoint Extraction for VRM Animation
Extracts high-quality pose + hand landmarks from sign language videos
"""
import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time

class KeypointExtractor:
    """Extract keypoints using MediaPipe Pose + Hands"""
    
    def __init__(self, model_complexity=2, min_confidence=0.7):
        """
        Initialize MediaPipe models
        
        Args:
            model_complexity: 0 (lite), 1 (full), 2 (heavy) - higher is more accurate
            min_confidence: Minimum detection confidence (0.0 to 1.0)
        """
        print(" Initializing MediaPipe...")
        
        # Pose detection (body landmarks)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        
        # Hand detection (hand landmarks)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,  # 0 or 1 for hands
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        
        print(" MediaPipe initialized")
    
    def extract_from_video(self, video_path, start_frame=1, end_frame=None):
        """
        Extract keypoints from video clip
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame (1-indexed)
            end_frame: Ending frame (1-indexed, None = end of video)
        
        Returns:
            keypoints: numpy array of shape (num_frames, 75, 3)
            metadata: dict with extraction info
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust frame indices (convert to 0-indexed)
        start_idx = max(0, start_frame - 1)
        end_idx = min(total_frames, end_frame if end_frame else total_frames)
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        keypoints = []
        frame_confidences = []
        frames_processed = 0
        frames_with_data = 0
        
        for frame_idx in range(start_idx, end_idx):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            pose_results = self.pose.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)
            
            # Extract landmarks
            frame_landmarks, confidence = self._extract_frame_landmarks(
                pose_results, hands_results
            )
            
            keypoints.append(frame_landmarks)
            frame_confidences.append(confidence)
            frames_processed += 1
            
            if confidence > 0:
                frames_with_data += 1
        
        cap.release()
        
        # Convert to numpy array
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Metadata
        metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'start_frame': start_frame,
            'end_frame': end_frame if end_frame else total_frames,
            'frames_extracted': frames_processed,
            'frames_with_data': frames_with_data,
            'avg_confidence': np.mean(frame_confidences),
            'shape': keypoints.shape,
            'completeness': frames_with_data / frames_processed if frames_processed > 0 else 0
        }
        
        return keypoints, metadata
    
    def _extract_frame_landmarks(self, pose_results, hands_results):
        """
        Extract and combine landmarks from pose and hands
        
        Returns:
            landmarks: numpy array of shape (75, 3)
            confidence: average confidence score
        """
        landmarks = np.zeros((75, 3), dtype=np.float32)
        confidences = []
        
        # Extract pose landmarks (33 landmarks)
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
                confidences.append(landmark.visibility)
        
        # Extract hand landmarks (21 per hand)
        left_hand_landmarks = None
        right_hand_landmarks = None
        
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            ):
                # Determine which hand
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                confidences.append(hand_score)
                
                # Extract landmarks
                hand_data = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ], dtype=np.float32)
                
                if hand_label == 'Left':
                    left_hand_landmarks = hand_data
                else:
                    right_hand_landmarks = hand_data
        
        # Add left hand (landmarks 33-53)
        if left_hand_landmarks is not None:
            landmarks[33:54] = left_hand_landmarks
        
        # Add right hand (landmarks 54-74)
        if right_hand_landmarks is not None:
            landmarks[54:75] = right_hand_landmarks
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return landmarks, avg_confidence
    
    def validate_keypoints(self, keypoints, metadata):
        """
        Validate extracted keypoints
        
        Returns:
            is_valid: bool
            quality_report: dict with quality metrics
        """
        quality_report = {
            'valid': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check shape
        if keypoints.shape[1:] != (75, 3):
            quality_report['valid'] = False
            quality_report['issues'].append(f"Invalid shape: {keypoints.shape}")
            return False, quality_report
        
        # Check for minimum frames
        if keypoints.shape[0] < 5:
            quality_report['valid'] = False
            quality_report['issues'].append(f"Too few frames: {keypoints.shape[0]}")
        
        # Check completeness
        completeness = metadata.get('completeness', 0)
        quality_report['metrics']['completeness'] = completeness
        if completeness < 0.5:
            quality_report['valid'] = False
            quality_report['issues'].append(f"Low completeness: {completeness:.2%}")
        
        # Check for zeros (missing data)
        zero_frames = np.all(keypoints == 0, axis=(1, 2))
        num_zero_frames = np.sum(zero_frames)
        quality_report['metrics']['zero_frames'] = int(num_zero_frames)
        quality_report['metrics']['zero_frame_ratio'] = float(num_zero_frames / len(keypoints))
        
        if num_zero_frames > len(keypoints) * 0.5:
            quality_report['valid'] = False
            quality_report['issues'].append(f"Too many zero frames: {num_zero_frames}/{len(keypoints)}")
        
        # Check coordinate ranges
        non_zero_data = keypoints[~zero_frames]
        if len(non_zero_data) > 0:
            x_range = [float(non_zero_data[:, :, 0].min()), float(non_zero_data[:, :, 0].max())]
            y_range = [float(non_zero_data[:, :, 1].min()), float(non_zero_data[:, :, 1].max())]
            z_range = [float(non_zero_data[:, :, 2].min()), float(non_zero_data[:, :, 2].max())]
            
            quality_report['metrics']['x_range'] = x_range
            quality_report['metrics']['y_range'] = y_range
            quality_report['metrics']['z_range'] = z_range
            
            # Check if ranges are reasonable (MediaPipe typically outputs [0, 1] for x,y)
            if x_range[0] < -0.5 or x_range[1] > 1.5:
                quality_report['issues'].append(f"Unusual X range: {x_range}")
            if y_range[0] < -0.5 or y_range[1] > 1.5:
                quality_report['issues'].append(f"Unusual Y range: {y_range}")
        
        # Check hand visibility
        left_hand_visible = np.any(keypoints[:, 33:54, :] != 0, axis=(1, 2))
        right_hand_visible = np.any(keypoints[:, 54:75, :] != 0, axis=(1, 2))
        both_hands_visible = left_hand_visible & right_hand_visible
        
        hand_visibility = float(np.sum(both_hands_visible) / len(keypoints))
        quality_report['metrics']['hand_visibility'] = hand_visibility
        
        if hand_visibility < 0.3:
            quality_report['issues'].append(f"Low hand visibility: {hand_visibility:.2%}")
        
        # Check confidence
        avg_confidence = metadata.get('avg_confidence', 0)
        quality_report['metrics']['avg_confidence'] = float(avg_confidence)
        
        if avg_confidence < 0.5:
            quality_report['issues'].append(f"Low confidence: {avg_confidence:.2f}")
        
        # Overall quality score
        quality_score = (
            completeness * 0.3 +
            (1 - quality_report['metrics']['zero_frame_ratio']) * 0.2 +
            hand_visibility * 0.3 +
            avg_confidence * 0.2
        )
        quality_report['metrics']['quality_score'] = float(quality_score)
        
        return quality_report['valid'], quality_report
    
    def close(self):
        """Release resources"""
        self.pose.close()
        self.hands.close()


def _reconstruct_entry_from_npy(npy_path, sign_name, video_id, video_dir):
    """
    Rebuild an extraction_results entry from an already-saved .npy file.
    Called when the .npy exists on disk but is missing from extraction_results.json.
    """
    kps = np.load(npy_path)                        # shape (T, 75, 3)
    T   = kps.shape[0]

    zero_mask        = np.all(kps == 0, axis=(1, 2))
    num_zero_frames  = int(np.sum(zero_mask))
    zero_frame_ratio = num_zero_frames / T if T > 0 else 0.0
    completeness     = 1.0 - zero_frame_ratio

    non_zero = kps[~zero_mask]
    if len(non_zero):
        x_range = [float(non_zero[:, :, 0].min()), float(non_zero[:, :, 0].max())]
        y_range = [float(non_zero[:, :, 1].min()), float(non_zero[:, :, 1].max())]
        z_range = [float(non_zero[:, :, 2].min()), float(non_zero[:, :, 2].max())]
    else:
        x_range = y_range = z_range = [0.0, 0.0]

    lh_vis          = np.any(kps[:, 33:54, :] != 0, axis=(1, 2))
    rh_vis          = np.any(kps[:, 54:75, :] != 0, axis=(1, 2))
    hand_visibility = float(np.mean(lh_vis & rh_vis))
    avg_confidence  = completeness   # not stored in .npy; use completeness as proxy

    quality_score = (
        completeness          * 0.3 +
        (1 - zero_frame_ratio) * 0.2 +
        hand_visibility       * 0.3 +
        avg_confidence        * 0.2
    )

    issues = []
    if y_range[1] > 1.5:
        issues.append(f"Unusual Y range: {y_range}")
    if hand_visibility < 0.3:
        issues.append(f"Low hand visibility: {hand_visibility:.2%}")

    return {
        'video_id': video_id,
        'file':     str(npy_path),
        'metadata': {
            'video_path':       str(Path(video_dir) / f'{video_id}.mp4'),
            'fps':              25.0,
            'start_frame':      1,
            'end_frame':        T,
            'frames_extracted': T,
            'frames_with_data': T - num_zero_frames,
            'avg_confidence':   avg_confidence,
            'shape':            list(kps.shape),
            'completeness':     completeness,
        },
        'quality': {
            'valid':   completeness > 0.5,
            'issues':  issues,
            'metrics': {
                'completeness':     completeness,
                'zero_frames':      num_zero_frames,
                'zero_frame_ratio': zero_frame_ratio,
                'x_range':          x_range,
                'y_range':          y_range,
                'z_range':          z_range,
                'hand_visibility':  hand_visibility,
                'avg_confidence':   avg_confidence,
                'quality_score':    quality_score,
            },
        },
    }


def extract_nslt_100(output_dir='keypoints_fresh', max_signs=None):
    print("="*70)
    print("KEYPOINT EXTRACTION - NSLT-100  (resume-safe)")
    print("="*70)

    # -----------------------------------------------------------------------
    # Load metadata
    # -----------------------------------------------------------------------
    print("\n Loading metadata...")
    with open('nslt_100.json') as f:
        nslt_data = json.load(f)
    with open('checkpoints/vocab.json') as f:
        vocab = json.load(f)

    print(f" Loaded {len(nslt_data)} video entries")
    print(f" Loaded {len(vocab)} sign classes")

    # Create output directories
    output_path = Path(output_dir)
    signs_path  = output_path / 'signs'
    output_path.mkdir(exist_ok=True)
    signs_path.mkdir(exist_ok=True)

    results_file = output_path / 'extraction_results.json'
    summary_file = output_path / 'summary.json'

    # -----------------------------------------------------------------------
    # Step 1: load existing JSON results (may be empty / stale)
    # -----------------------------------------------------------------------
    if results_file.exists():
        with open(results_file) as f:
            extraction_results = json.load(f)
        print(f" Loaded previous results ({len(extraction_results)} sign keys)")
    else:
        extraction_results = {}

    # -----------------------------------------------------------------------
    # Step 2: scan .npy files on disk — recover entries missing from JSON
    # -----------------------------------------------------------------------
    print("\n Scanning existing .npy files on disk...")

    # Build lookup: (sign_name, video_id) -> Path
    disk_npy = {}
    for npy_path in sorted(signs_path.glob('*.npy')):
        parts = npy_path.stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            disk_npy[(parts[0], parts[1])] = npy_path

    print(f" Found {len(disk_npy)} .npy files on disk")

    # Build set of (sign, video_id) already in JSON
    in_json = {
        (sign_name, item['video_id'])
        for sign_name, entries in extraction_results.items()
        for item in entries
    }

    # Recover entries that are on disk but absent from JSON
    recovered = 0
    for (sign_name, video_id), npy_path in disk_npy.items():
        if (sign_name, video_id) not in in_json:
            extraction_results.setdefault(sign_name, [])
            try:
                entry = _reconstruct_entry_from_npy(
                    npy_path, sign_name, video_id,
                    video_dir='previous/videos'
                )
                extraction_results[sign_name].append(entry)
                recovered += 1
            except Exception as e:
                print(f"   WARNING: could not recover {npy_path.name}: {e}")

    if recovered:
        print(f" Recovered {recovered} entries from disk (not in JSON)")

    # already_done = every (sign, video_id) now accounted for
    already_done = {
        (sign_name, item['video_id'])
        for sign_name, entries in extraction_results.items()
        for item in entries
    }
    print(f" Already done: {len(already_done)} videos — will skip these")

    # -----------------------------------------------------------------------
    # Step 3: build the full extraction plan
    # -----------------------------------------------------------------------
    signs_data = {}
    for video_id, info in nslt_data.items():
        sign_name = vocab[str(info['action'][0])]
        signs_data.setdefault(sign_name, []).append({
            'video_id':    video_id,
            'start_frame': info['action'][1],
            'end_frame':   info['action'][2],
            'subset':      info['subset'],
        })

    print(f"\n Found {len(signs_data)} unique signs")

    if max_signs:
        signs_data = dict(list(signs_data.items())[:max_signs])
        print(f" Limited to {max_signs} signs for testing")

    # Count pending
    pending_total = sum(
        1 for sign_name, instances in signs_data.items()
        for inst in instances
        if (sign_name, inst['video_id']) not in already_done
    )
    print(f" Pending extraction: {pending_total} videos")

    if pending_total == 0:
        print("\n Nothing left to extract — saving results and exiting.")
        _save_results(extraction_results, signs_data, results_file, summary_file,
                      successful=recovered, failed=0,
                      skipped=len(already_done) - recovered)
        return extraction_results, _load_json(summary_file)

    # -----------------------------------------------------------------------
    # Step 4: extract only the missing videos
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("EXTRACTING REMAINING VIDEOS")
    print("="*70)

    extractor  = KeypointExtractor(model_complexity=2, min_confidence=0.7)
    successful = 0
    failed     = 0

    for sign_name in sorted(signs_data.keys()):
        instances = signs_data[sign_name]

        pending_for_sign = [
            inst for inst in instances
            if (sign_name, inst['video_id']) not in already_done
        ]
        if not pending_for_sign:
            continue   # entire sign already done — skip silently

        print(f"\n {sign_name}  ({len(pending_for_sign)} remaining / {len(instances)} total)")
        extraction_results.setdefault(sign_name, [])

        for inst in pending_for_sign:
            video_id    = inst['video_id']
            video_path  = Path('previous/videos') / f'{video_id}.mp4'
            output_file = signs_path / f'{sign_name}_{video_id}.npy'

            if not video_path.exists():
                print(f"   {video_id}: video not found — skip")
                failed += 1
                continue

            try:
                keypoints, metadata = extractor.extract_from_video(
                    video_path,
                    start_frame=inst['start_frame'],
                    end_frame=inst['end_frame'],
                )
                is_valid, quality_report = extractor.validate_keypoints(keypoints, metadata)

                if is_valid:
                    np.save(output_file, keypoints)
                    quality_score = quality_report['metrics']['quality_score']
                    print(f"   {video_id}: {keypoints.shape[0]} frames  quality={quality_score:.2f}")

                    extraction_results[sign_name].append({
                        'video_id': video_id,
                        'file':     str(output_file),
                        'metadata': metadata,
                        'quality':  quality_report,
                    })
                    already_done.add((sign_name, video_id))
                    successful += 1
                else:
                    issues = ', '.join(quality_report['issues'])
                    print(f"   {video_id}: invalid — {issues}")
                    failed += 1

            except Exception as e:
                print(f"   {video_id}: error — {e}")
                failed += 1

        # Save after every sign so progress survives interruption
        _save_results(extraction_results, signs_data, results_file, summary_file,
                      successful=recovered + successful,
                      failed=failed,
                      skipped=len(already_done) - recovered - successful,
                      in_progress=True)

    extractor.close()

    # Final save
    _save_results(extraction_results, signs_data, results_file, summary_file,
                  successful=recovered + successful,
                  failed=failed,
                  skipped=len(already_done) - recovered - successful,
                  in_progress=False)

    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f" Recovered from disk : {recovered}")
    print(f" Newly extracted     : {successful}")
    print(f" Skipped (done)      : {len(already_done) - recovered - successful}")
    print(f" Failed / not found  : {failed}")
    print(f" Results             : {results_file}")
    print(f" Summary             : {summary_file}")
    print("="*70)

    return extraction_results, _load_json(summary_file)


def _save_results(extraction_results, signs_data, results_file, summary_file,
                  successful, failed, skipped, in_progress=False):
    """Write extraction_results.json and summary.json to disk."""
    with open(results_file, 'w') as f:
        json.dump(extraction_results, f, indent=2)

    summary = {
        'total_signs':            len(signs_data),
        'successful_extractions': successful,
        'failed_extractions':     failed,
        'skipped':                skipped,
        'output_directory':       str(results_file.parent),
        'timestamp':              time.strftime('%Y-%m-%d %H:%M:%S'),
        'status':                 'in_progress' if in_progress else 'complete',
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    print("\n Starting prototype extraction (5 signs)...")
    print("This will test the pipeline before full extraction.\n")

    results, summary = extract_nslt_100(
        output_dir='keypoints_fresh',
        max_signs=5
    )

    print("\n Prototype complete!")
    print("\nNext steps:")
    print("1. Check keypoints_fresh/signs/ for extracted files")
    print("2. Review extraction_results.json for quality metrics")
    print("3. If satisfied, run full extraction:")
    print("   python previous/Prev_2/extract_all_100_signs.py")
