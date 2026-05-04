"""
wlasl_service.py - WLASL-100 ASL Recognition Service

Provides sign-to-text translation using the WLASL-100 model with:
- MediaPipe Holistic keypoint extraction
- TCN + BiGRU model inference
- Test-Time Augmentation (TTA) for robustness
- Temperature scaling for calibrated confidence scores

Performance notes:
- Video extraction runs in a thread executor to avoid blocking the async event loop
- Frames are processed one at a time (no full-video RAM load)
- Resampling uses nearest-neighbor to match training-time dataset.py behaviour
- TTA is enabled by default for uploaded videos; disable for real-time use
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

from .wlasl_model import SignLanguageTransformer
from .keypoint_extractor import KeypointExtractor

# Thread pool for CPU-bound / blocking work (MediaPipe + OpenCV)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wlasl_worker")


logger = logging.getLogger(__name__)


class WLASLModelService:
    """
    Service for WLASL-100 model inference.
    
    Handles:
    - Model loading and initialization
    - Video processing and keypoint extraction
    - Sequence preprocessing and resampling
    - Test-Time Augmentation (TTA)
    - Inference with temperature scaling
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        temperature_path: Optional[str] = None,
        device: str = "cpu",
        use_tta: bool = True,
        confidence_threshold: float = 0.25,
        num_frames: int = 64,
    ):
        """
        Initialize WLASL model service.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            vocab_path: Path to vocabulary JSON file
            temperature_path: Path to temperature JSON file (optional)
            device: Device to run model on ('cuda' or 'cpu')
            use_tta: Enable Test-Time Augmentation
            confidence_threshold: Minimum confidence for predictions
            num_frames: Target sequence length (default: 64)
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.temperature_path = temperature_path
        self.device = device
        self.use_tta = use_tta
        self.confidence_threshold = confidence_threshold
        self.num_frames = num_frames
        
        # Model and vocabulary (loaded on startup)
        self.model: Optional[SignLanguageTransformer] = None
        self.vocab: Optional[Dict[str, str]] = None
        self.temperature: float = 1.0
        
        # Keypoint extractor
        self.extractor: Optional[KeypointExtractor] = None
        
        # Statistics
        self.total_predictions = 0
        self.total_inference_time_ms = 0.0
        self.total_confidence = 0.0
        
        logger.info(f"WLASLModelService initialized with device={device}, use_tta={use_tta}")
    
    async def load_model(self):
        """Load model, vocabulary, and temperature on startup."""
        try:
            # Verify files exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            if not os.path.exists(self.vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
            
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = SignLanguageTransformer.load_from_checkpoint(
                self.model_path,
                device=self.device
            )
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Load vocabulary
            logger.info(f"Loading vocabulary from {self.vocab_path}")
            with open(self.vocab_path, 'r') as f:
                self.vocab = json.load(f)
            logger.info(f"Vocabulary loaded: {len(self.vocab)} words")
            
            # Load temperature (optional)
            if self.temperature_path and os.path.exists(self.temperature_path):
                with open(self.temperature_path, 'r') as f:
                    temp_data = json.load(f)
                    self.temperature = float(temp_data.get("temperature", 1.0))
                logger.info(f"Loaded calibrated temperature: {self.temperature}")
            else:
                logger.info("Using default temperature: 1.0")
            
            # Initialize keypoint extractor
            # model_complexity=1 matches 1_extract_keypoints.py used during training
            self.extractor = KeypointExtractor(model_complexity=1)
            logger.info("KeypointExtractor initialized (model_complexity=1)")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def resample_sequence(
        self,
        keypoints: np.ndarray,
        target_frames: int = 64
    ) -> np.ndarray:
        """
        Resample keypoint sequence to target length using nearest-neighbor sampling.

        Matches the behaviour of dataset.py used during training (np.linspace + int cast),
        which avoids a distribution shift between training and inference.

        Args:
            keypoints: Input sequence of shape (num_frames, 126)
            target_frames: Target sequence length (default: 64)
        Returns:
            Resampled sequence of shape (target_frames, 126)
        """
        num_frames = keypoints.shape[0]

        if num_frames == target_frames:
            return keypoints

        if num_frames < target_frames:
            # Pad by repeating last frame (same as training-time padding)
            padding = np.repeat(
                keypoints[-1:],
                target_frames - num_frames,
                axis=0
            )
            return np.vstack([keypoints, padding])

        # Nearest-neighbor resampling — matches training dataset.py
        indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
        return keypoints[indices]
    
    def build_tta_variants(self, seq: np.ndarray) -> List[np.ndarray]:
        """
        Generate Test-Time Augmentation variants.
        
        Creates 4 variants:
        1. Center sample (standard)
        2. Speed-up (first 85% of frames)
        3. Slow-down (last 85% of frames)
        4. Horizontal mirror (swap hands, flip x coordinates)
        
        Args:
            seq: Input sequence of shape (num_frames, 126)
        Returns:
            List of 4 sequences, each of shape (target_frames, 126)
        """
        T = seq.shape[0]
        variants = []
        
        # 1. Center sample
        indices = np.linspace(0, T - 1, self.num_frames).astype(int)
        variants.append(seq[indices])
        
        # 2. Speed-up (first 85%)
        end = max(int(T * 0.85), self.num_frames)
        indices = np.linspace(0, end - 1, self.num_frames).astype(int)
        variants.append(seq[indices])
        
        # 3. Slow-down (last 85%)
        start = min(int(T * 0.15), T - self.num_frames)
        indices = np.linspace(start, T - 1, self.num_frames).astype(int)
        variants.append(seq[indices])
        
        # 4. Horizontal mirror (swap hands, flip x)
        mirrored = seq.copy()
        lh = mirrored[:, :63].copy()
        rh = mirrored[:, 63:126].copy()
        
        # Flip x coordinates (every 3rd element starting at 0)
        lh[:, 0::3] = 1.0 - lh[:, 0::3]
        rh[:, 0::3] = 1.0 - rh[:, 0::3]
        
        # Swap hands
        mirrored[:, :63] = rh
        mirrored[:, 63:126] = lh
        
        indices = np.linspace(0, T - 1, self.num_frames).astype(int)
        variants.append(mirrored[indices])
        
        return variants
    
    async def extract_keypoints_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract keypoints from video file without loading all frames into RAM.

        Frames are decoded and processed one at a time inside a thread executor
        so the async event loop is never blocked by OpenCV or MediaPipe.

        Args:
            video_path: Path to video file
        Returns:
            Keypoint sequence of shape (num_frames, 126)
        Raises:
            ValueError: If video cannot be opened or processed
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        def _extract_sync() -> np.ndarray:
            """Blocking extraction — runs in thread pool.
            
            Creates its own KeypointExtractor instance because MediaPipe's
            Holistic object is NOT thread-safe. Sharing one across threads
            produces corrupted keypoints and random predictions.
            
            Uses static_image_mode=True for uploaded videos — re-detects
            hands every frame instead of tracking, which is more reliable
            for pre-recorded video files.
            """
            # Each thread gets its own extractor — safe for concurrent requests
            # static_image_mode=False: tracking mode works better for continuous video recordings
            local_extractor = KeypointExtractor(model_complexity=1, static_image_mode=False)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                local_extractor.close()
                raise ValueError(f"Cannot open video file: {video_path}")

            keypoints_list = []
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        kp = local_extractor.extract_keypoints_from_frame(frame)
                        keypoints_list.append(kp)
                    except Exception as e:
                        logger.warning(f"Frame {frame_idx} extraction failed: {e}")
                        keypoints_list.append(np.zeros(126, dtype=np.float32))
                    frame_idx += 1
            finally:
                cap.release()
                local_extractor.close()

            if len(keypoints_list) == 0:
                raise ValueError("Video contains no frames")

            logger.info(f"Extracted keypoints from {frame_idx} frames")
            return np.array(keypoints_list, dtype=np.float32)

        loop = asyncio.get_event_loop()
        keypoints = await loop.run_in_executor(_executor, _extract_sync)
        return keypoints
    
    async def predict(
        self,
        keypoints: np.ndarray,
        use_tta: Optional[bool] = None
    ) -> Dict:
        """
        Run inference on keypoint sequence.

        Inference runs in a thread executor so the async event loop is not
        blocked by PyTorch CPU computation.

        Args:
            keypoints: Keypoint sequence of shape (num_frames, 126)
            use_tta: Override default TTA setting (optional).
                     Set False for real-time/WebSocket use to reduce latency.
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()
        use_tta = use_tta if use_tta is not None else self.use_tta

        # Resample to target length
        keypoints = self.resample_sequence(keypoints, self.num_frames)

        # Generate TTA variants if enabled
        if use_tta:
            variants = self.build_tta_variants(keypoints)
        else:
            variants = [keypoints]

        # Convert to tensor and batch
        batch = np.stack(variants, axis=0)  # (num_variants, num_frames, 126)
        x = torch.from_numpy(batch).to(self.device)

        def _infer_sync():
            # Pure float32 — no autocast. The model was trained in float32 and
            # autocast on CPU in PyTorch 2.x defaults to bfloat16, which produces
            # completely different logits and causes random/wrong predictions.
            with torch.no_grad():
                return self.model(x)  # (num_variants, num_classes)

        loop = asyncio.get_event_loop()
        logits = await loop.run_in_executor(_executor, _infer_sync)

        # Average logits across variants
        avg_logits = logits.mean(dim=0, keepdim=True)  # (1, num_classes)

        # Apply temperature scaling
        scaled_logits = avg_logits / self.temperature

        # Compute probabilities
        probs = torch.softmax(scaled_logits, dim=-1)[0].cpu().numpy()

        # Get top 5 predictions
        top5_indices = np.argsort(probs)[::-1][:5]
        top5_predictions = [
            (int(idx), float(probs[idx]))
            for idx in top5_indices
        ]

        # Get glosses
        glosses = [
            self.vocab.get(str(idx), self.vocab.get(idx, f"unknown_{idx}"))
            for idx, _ in top5_predictions
        ]

        # Processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update statistics
        self.total_predictions += 1
        self.total_inference_time_ms += processing_time_ms
        self.total_confidence += top5_predictions[0][1]

        logger.info(
            f"Inference completed: {glosses[0]} "
            f"(confidence={top5_predictions[0][1]:.3f}, "
            f"time={processing_time_ms}ms, tta={use_tta})"
        )

        return {
            "recognized_text": glosses[0],
            "glosses": glosses,
            "confidence": top5_predictions[0][1],
            "top5_predictions": top5_predictions,
            "frame_count": len(keypoints),
            "processing_time_ms": processing_time_ms,
            "sign_language": "ASL",
        }
    
    async def sign_to_text(
        self,
        video_path: str,
        use_tta: Optional[bool] = None
    ) -> Dict:
        """
        Main entry point for sign-to-text translation.
        
        Args:
            video_path: Path to video file
            use_tta: Override default TTA setting (optional)
        Returns:
            Dictionary with translation results
        """
        try:
            # Extract keypoints
            keypoints = await self.extract_keypoints_from_video(video_path)
            
            # Run inference
            result = await self.predict(keypoints, use_tta=use_tta)
            
            return result
            
        except Exception as e:
            logger.error(f"Sign-to-text failed: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get service statistics."""
        avg_inference_time = (
            self.total_inference_time_ms / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        avg_confidence = (
            self.total_confidence / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        
        return {
            "total_predictions": self.total_predictions,
            "average_inference_time_ms": round(avg_inference_time, 2),
            "average_confidence": round(avg_confidence, 3),
            "model_loaded": self.model is not None,
            "device": self.device,
        }
    
    def close(self):
        """Release resources."""
        if self.extractor:
            self.extractor.close()
        logger.info("WLASLModelService closed")


# Singleton instance (initialized on startup)
_wlasl_service: Optional[WLASLModelService] = None


async def get_wlasl_service() -> WLASLModelService:
    """Get or create WLASL service singleton."""
    global _wlasl_service
    
    if _wlasl_service is None:
        # Get configuration from environment or use defaults
        model_path = os.getenv(
            "MODEL_PATH",
            "app/ml/models/wlasl100/best_model.pth"
        )
        vocab_path = os.getenv(
            "VOCAB_PATH",
            "app/ml/models/wlasl100/vocab.json"
        )
        temperature_path = os.getenv(
            "TEMPERATURE_PATH",
            "app/ml/models/wlasl100/temperature.json"
        )
        # Always use CPU — GPU not available in Docker container
        device = os.getenv("ML_DEVICE", "cpu")
        use_tta = os.getenv("USE_TTA", "true").lower() == "true"
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
        
        _wlasl_service = WLASLModelService(
            model_path=model_path,
            vocab_path=vocab_path,
            temperature_path=temperature_path,
            device=device,
            use_tta=use_tta,
            confidence_threshold=confidence_threshold,
        )
        
        await _wlasl_service.load_model()
    
    return _wlasl_service


if __name__ == "__main__":
    # Test the service
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        service = WLASLModelService(
            model_path="app/ml/models/wlasl100/best_model.pth",
            vocab_path="app/ml/models/wlasl100/vocab.json",
            device="cpu",
        )
        await service.load_model()
        
        # Test with dummy keypoints
        dummy_keypoints = np.random.randn(100, 126).astype(np.float32)
        result = await service.predict(dummy_keypoints)
        
        print("Prediction result:")
        print(f"  Recognized text: {result['recognized_text']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Top 5: {result['glosses']}")
        print(f"  Processing time: {result['processing_time_ms']}ms")
        
        service.close()
    
    asyncio.run(test())
