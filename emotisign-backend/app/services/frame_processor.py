"""
Frame Processor Service
=======================
Handles incoming base64 webcam frames for real-time sign recognition.

Buffer Strategy:
  - Collects N frames into a sliding window
  - Runs sign detection on the window every PROCESS_EVERY_N frames
  - Returns partial results immediately so the client sees live feedback

Implementation:
  - Decodes base64 frames → numpy arrays via OpenCV
  - Extracts MediaPipe hand keypoints per frame (in thread executor)
  - Resamples sequence to 64 frames (nearest-neighbor, matches training)
  - Runs TCN+BiGRU inference WITHOUT TTA for low-latency real-time use
  - Face emotion (happy / sad / neutral) via MediaPipe Face Mesh
"""

import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── Constants ──
FRAME_BUFFER_SIZE = 30       # sliding window of N frames
PROCESS_EVERY_N  = 10        # run inference every N received frames
MIN_FRAMES_TO_PROCESS = 5   # don't bother below this

EMOTIONS = ["happy", "neutral", "sad"]


@dataclass
class ProcessingResult:
    recognized_text: str
    glosses: list
    confidence: float
    emotion: str
    emotion_scores: dict
    is_partial: bool          # True = still processing, False = confident result
    frame_count: int
    processing_time_ms: int


@dataclass
class FrameBuffer:
    """Per-session sliding frame buffer"""
    frames: deque = field(default_factory=lambda: deque(maxlen=FRAME_BUFFER_SIZE))
    total_received: int = 0
    last_result: Optional[ProcessingResult] = None
    session_start: float = field(default_factory=time.time)

    def add_frame(self, frame_b64: str):
        self.frames.append(frame_b64)
        self.total_received += 1

    def should_process(self) -> bool:
        return (
            len(self.frames) >= MIN_FRAMES_TO_PROCESS and
            self.total_received % PROCESS_EVERY_N == 0
        )

    def get_frames(self) -> list:
        return list(self.frames)


def _decode_frames_and_extract_keypoints(frames_b64: list) -> Optional[np.ndarray]:
    """
    Blocking function — runs in thread executor.

    Decodes base64 frames, extracts MediaPipe hand keypoints from each,
    and returns a (N, 126) float32 array.

    Returns None if no valid frames could be decoded.
    """
    # Import here to avoid circular imports at module load time
    try:
        from app.ml.keypoint_extractor import KeypointExtractor
    except ImportError:
        return None

    # Use a fresh extractor per call (stateless for real-time frames)
    # model_complexity=1 matches training extraction
    try:
        extractor = KeypointExtractor(model_complexity=1)
    except Exception:
        return None

    keypoints_list = []
    try:
        for frame_b64 in frames_b64:
            try:
                # Strip data URI prefix if present
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]
                raw = base64.b64decode(frame_b64)
                arr = np.frombuffer(raw, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    keypoints_list.append(np.zeros(126, dtype=np.float32))
                    continue
                kp = extractor.extract_keypoints_from_frame(img)
                keypoints_list.append(kp)
            except Exception:
                keypoints_list.append(np.zeros(126, dtype=np.float32))
    finally:
        extractor.close()

    if not keypoints_list:
        return None

    return np.array(keypoints_list, dtype=np.float32)


def _decode_middle_frame_b64(frames_b64: list) -> Optional[np.ndarray]:
    """Decode the middle frame from a base64 buffer for face emotion."""
    if not frames_b64:
        return None
    mid = frames_b64[len(frames_b64) // 2]
    try:
        if "," in mid:
            mid = mid.split(",", 1)[1]
        raw = base64.b64decode(mid)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _detect_face_emotion(frame_bgr: Optional[np.ndarray]) -> tuple[str, dict]:
    if frame_bgr is None:
        return "neutral", {"happy": 0.1, "sad": 0.1, "neutral": 0.7}
    try:
        from app.ml.face_emotion_service import get_face_emotion_detector

        det = get_face_emotion_detector().detect_from_frame(frame_bgr)
        return det["emotion"], det["emotion_scores"]
    except Exception:
        return "neutral", {"happy": 0.1, "sad": 0.1, "neutral": 0.7}


def _run_inference(keypoints: np.ndarray) -> Optional[dict]:
    """
    Blocking function — runs in thread executor.

    Resamples keypoints to 64 frames and runs TCN+BiGRU inference
    WITHOUT TTA (use_tta=False) for low-latency real-time use.

    Returns the prediction dict from WLASLModelService, or None on failure.
    """
    import asyncio as _asyncio

    try:
        from app.ml.wlasl_service import _wlasl_service
        if _wlasl_service is None or _wlasl_service.model is None:
            return None

        # Resample — nearest-neighbor to match training
        num_frames = keypoints.shape[0]
        target = _wlasl_service.num_frames
        if num_frames != target:
            if num_frames < target:
                padding = np.repeat(keypoints[-1:], target - num_frames, axis=0)
                keypoints = np.vstack([keypoints, padding])
            else:
                indices = np.linspace(0, num_frames - 1, target).astype(int)
                keypoints = keypoints[indices]

        import torch
        x = torch.from_numpy(keypoints[np.newaxis]).to(_wlasl_service.device)  # (1, T, 126)

        with torch.no_grad():
            logits = _wlasl_service.model(x)  # (1, num_classes)

        scaled = logits / _wlasl_service.temperature
        probs = torch.softmax(scaled, dim=-1)[0].cpu().numpy()

        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(int(i), float(probs[i])) for i in top5_idx]
        glosses = [
            _wlasl_service.vocab.get(str(i), _wlasl_service.vocab.get(i, f"#{i}"))
            for i, _ in top5
        ]

        return {
            "recognized_text": glosses[0],
            "glosses": glosses,
            "confidence": top5[0][1],
        }
    except Exception:
        return None


async def process_frame_buffer(buffer: FrameBuffer) -> ProcessingResult:
    """
    Core processing function — runs on each trigger.

    1. Decodes base64 frames and extracts MediaPipe keypoints (thread executor)
    2. Runs TCN+BiGRU inference without TTA for low latency (thread executor)
    3. Falls back to a low-confidence placeholder if the model is unavailable
    """
    start = time.time()
    frames_b64 = buffer.get_frames()

    loop = asyncio.get_event_loop()

    # Step 1: decode + extract keypoints (blocking, offloaded)
    keypoints = await loop.run_in_executor(
        None, _decode_frames_and_extract_keypoints, frames_b64
    )

    result_dict = None
    if keypoints is not None and keypoints.shape[0] >= 2:
        # Step 2: inference (blocking, offloaded)
        result_dict = await loop.run_in_executor(None, _run_inference, keypoints)

    processing_time = int((time.time() - start) * 1000)

    middle_frame = await loop.run_in_executor(
        None, _decode_middle_frame_b64, frames_b64
    )
    dominant_emotion, emotion_scores = await loop.run_in_executor(
        None, _detect_face_emotion, middle_frame
    )

    if result_dict is not None:
        return ProcessingResult(
            recognized_text=result_dict["recognized_text"],
            glosses=result_dict["glosses"],
            confidence=result_dict["confidence"],
            emotion=dominant_emotion,
            emotion_scores=emotion_scores,
            is_partial=result_dict["confidence"] < 0.40,
            frame_count=len(buffer.frames),
            processing_time_ms=processing_time,
        )

    # Fallback: model not loaded or extraction failed
    return ProcessingResult(
        recognized_text="",
        glosses=[],
        confidence=0.0,
        emotion=dominant_emotion,
        emotion_scores=emotion_scores,
        is_partial=True,
        frame_count=len(buffer.frames),
        processing_time_ms=processing_time,
    )


async def decode_base64_frame(frame_b64: str) -> Optional[bytes]:
    """
    Decode a base64 encoded image frame.
    Strips data URI prefix if present (e.g. 'data:image/jpeg;base64,...')
    """
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        return base64.b64decode(frame_b64)
    except Exception:
        return None
