"""
Frame Processor Service
=======================
Handles incoming base64 webcam frames for real-time sign recognition.

Buffer Strategy:
  - Collects N frames into a sliding window
  - Runs sign detection on the window every PROCESS_EVERY_N frames
  - Returns partial results immediately so the client sees live feedback

REAL IMPLEMENTATION NOTES:
  - Replace _run_sign_detection() with your actual MediaPipe + model inference
  - Replace _run_emotion_detection() with DeepFace / FER on the face crop
  - For production: offload heavy inference to a thread pool (run_in_executor)
    so it doesn't block the async event loop
"""

import asyncio
import base64
import io
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── Constants ──
FRAME_BUFFER_SIZE = 30       # sliding window of N frames
PROCESS_EVERY_N  = 10        # run inference every N received frames
MIN_FRAMES_TO_PROCESS = 5   # don't bother below this

MOCK_GLOSSES = [
    ["HELLO", "HOW", "YOU"],
    ["THANK", "YOU"],
    ["PLEASE", "HELP", "ME"],
    ["I", "LOVE", "SIGN", "LANGUAGE"],
    ["GOOD", "MORNING"],
    ["MY", "NAME", "IS"],
    ["NICE", "MEET", "YOU"],
    ["UNDERSTAND", "YOU"],
]

MOCK_SENTENCES = [
    "Hello, how are you?",
    "Thank you very much.",
    "Please help me.",
    "I love sign language.",
    "Good morning!",
    "What is your name?",
    "Nice to meet you.",
    "Do you understand me?",
]

EMOTIONS = ["happy", "neutral", "sad", "angry", "surprised"]


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


async def process_frame_buffer(buffer: FrameBuffer) -> ProcessingResult:
    """
    Core processing function — runs on each trigger.

    REAL IMPLEMENTATION:
      1. Decode base64 frames → numpy arrays (cv2.imdecode)
      2. Run MediaPipe Holistic on each frame → extract landmarks
      3. Stack landmark sequences → shape (N, 1662) or similar
      4. Normalize sequence length (pad/trim to fixed window)
      5. Run through trained LSTM/Transformer model → gloss predictions
      6. Convert glosses → natural language (seq2seq or lookup)
      7. Run emotion detection on face crop of middle frame
      8. Return structured result
    """
    start = time.time()

    # Simulate async inference (replace with actual model call)
    await asyncio.sleep(0.05)

    # ── PLACEHOLDER: mock sign recognition ──
    idx = random.randint(0, len(MOCK_GLOSSES) - 1)
    glosses = MOCK_GLOSSES[idx]
    text = MOCK_SENTENCES[idx]
    confidence = round(random.uniform(0.65, 0.97), 3)

    # ── PLACEHOLDER: mock emotion detection ──
    dominant_emotion = random.choice(EMOTIONS)
    emotion_scores = {e: round(random.uniform(0.01, 0.15), 3) for e in EMOTIONS}
    emotion_scores[dominant_emotion] = round(random.uniform(0.5, 0.88), 3)

    processing_time = int((time.time() - start) * 1000)

    return ProcessingResult(
        recognized_text=text,
        glosses=glosses,
        confidence=confidence,
        emotion=dominant_emotion,
        emotion_scores=emotion_scores,
        is_partial=confidence < 0.80,
        frame_count=len(buffer.frames),
        processing_time_ms=processing_time
    )


async def decode_base64_frame(frame_b64: str) -> Optional[bytes]:
    """
    Decode a base64 encoded image frame.
    Strips data URI prefix if present (e.g. 'data:image/jpeg;base64,...')

    REAL IMPLEMENTATION:
      After decoding, convert to numpy array:
        import numpy as np, cv2
        arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    """
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        return base64.b64decode(frame_b64)
    except Exception:
        return None
