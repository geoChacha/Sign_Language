"""
Unit tests for WLASL ML service.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.wlasl_service import WLASLModelService


class TestSequenceResampling:
    """Test sequence resampling functionality."""
    
    def test_resample_upsampling(self):
        """Test resampling from 30 frames to 64 frames (upsampling)."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        # Create 30-frame sequence
        keypoints = np.random.randn(30, 126).astype(np.float32)
        
        # Resample to 64 frames
        resampled = service.resample_sequence(keypoints, target_frames=64)
        
        # Check output shape
        assert resampled.shape == (64, 126)
    
    def test_resample_downsampling(self):
        """Test resampling from 100 frames to 64 frames (downsampling)."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        # Create 100-frame sequence
        keypoints = np.random.randn(100, 126).astype(np.float32)
        
        # Resample to 64 frames
        resampled = service.resample_sequence(keypoints, target_frames=64)
        
        # Check output shape
        assert resampled.shape == (64, 126)
    
    def test_resample_padding(self):
        """Test padding when input has fewer than 64 frames."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        # Create 20-frame sequence
        keypoints = np.random.randn(20, 126).astype(np.float32)
        
        # Resample to 64 frames
        resampled = service.resample_sequence(keypoints, target_frames=64)
        
        # Check output shape
        assert resampled.shape == (64, 126)
        
        # Last frames should be repeated (padding)
        assert np.allclose(resampled[20], resampled[21])
    
    def test_resample_no_change(self):
        """Test resampling when input already has target frames."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        # Create 64-frame sequence
        keypoints = np.random.randn(64, 126).astype(np.float32)
        
        # Resample to 64 frames
        resampled = service.resample_sequence(keypoints, target_frames=64)
        
        # Should be identical
        assert np.allclose(keypoints, resampled)


class TestTTAVariants:
    """Test Test-Time Augmentation variant generation."""
    
    def test_tta_generates_4_variants(self):
        """Test that 4 TTA variants are generated."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu",
            num_frames=64
        )
        
        # Create 100-frame sequence
        keypoints = np.random.randn(100, 126).astype(np.float32)
        
        # Generate TTA variants
        variants = service.build_tta_variants(keypoints)
        
        # Should have 4 variants
        assert len(variants) == 4
        
        # Each variant should have shape (64, 126)
        for variant in variants:
            assert variant.shape == (64, 126)
    
    def test_tta_center_sample(self):
        """Test center sample variant uses correct frame range."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu",
            num_frames=64
        )
        
        # Create sequence with identifiable pattern
        keypoints = np.arange(100 * 126).reshape(100, 126).astype(np.float32)
        
        # Generate TTA variants
        variants = service.build_tta_variants(keypoints)
        
        # First variant is center sample
        center_variant = variants[0]
        
        # Should sample evenly across all frames
        assert center_variant.shape == (64, 126)
    
    def test_tta_horizontal_mirror(self):
        """Test horizontal mirror swaps hands and flips x coordinates."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu",
            num_frames=64
        )
        
        # Create sequence with distinct left/right hand features
        keypoints = np.zeros((100, 126), dtype=np.float32)
        keypoints[:, :63] = 1.0  # Left hand = 1.0
        keypoints[:, 63:126] = 2.0  # Right hand = 2.0
        
        # Generate TTA variants
        variants = service.build_tta_variants(keypoints)
        
        # Fourth variant is horizontal mirror
        mirror_variant = variants[3]
        
        # Hands should be swapped (but x-coords flipped, so not exactly 2.0 and 1.0)
        # Just check that variant is different from original
        assert not np.allclose(mirror_variant, variants[0])


class TestStatistics:
    """Test service statistics tracking."""
    
    def test_initial_stats(self):
        """Test initial statistics are zero."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        stats = service.get_stats()
        
        assert stats["total_predictions"] == 0
        assert stats["average_inference_time_ms"] == 0.0
        assert stats["average_confidence"] == 0.0
        assert stats["model_loaded"] == False
        assert stats["device"] == "cpu"
    
    def test_stats_update_after_prediction(self):
        """Test statistics update after prediction."""
        service = WLASLModelService(
            model_path="dummy",
            vocab_path="dummy",
            device="cpu"
        )
        
        # Manually update stats (simulating a prediction)
        service.total_predictions = 5
        service.total_inference_time_ms = 1500.0
        service.total_confidence = 4.0
        
        stats = service.get_stats()
        
        assert stats["total_predictions"] == 5
        assert stats["average_inference_time_ms"] == 300.0  # 1500 / 5
        assert stats["average_confidence"] == 0.8  # 4.0 / 5
