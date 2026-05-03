"""
Unit tests for WLASL model architecture.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.wlasl_model import SignLanguageTransformer


class TestModelArchitecture:
    """Test model initialization and forward pass."""
    
    def test_model_initialization(self):
        """Test model initializes with correct dimensions."""
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        
        assert model is not None
        assert model.feature_dim == 126
        assert model.num_classes == 100
        assert model.d_model == 192
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        model.eval()
        
        # Create dummy input (batch_size=1, seq_len=64, features=126)
        x = torch.randn(1, 64, 126)
        
        # Forward pass
        output = model(x)
        
        # Check output shape (batch_size=1, num_classes=100)
        assert output.shape == (1, 100)
    
    def test_forward_pass_batch(self):
        """Test forward pass with batch size > 1."""
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        model.eval()
        
        # Create dummy input (batch_size=4, seq_len=64, features=126)
        x = torch.randn(4, 64, 126)
        
        # Forward pass
        output = model(x)
        
        # Check output shape (batch_size=4, num_classes=100)
        assert output.shape == (4, 100)
    
    def test_dropout_disabled_in_eval_mode(self):
        """Test dropout is disabled in evaluation mode."""
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 64, 126)
        
        # Run forward pass twice
        output1 = model(x)
        output2 = model(x)
        
        # Outputs should be identical (dropout disabled)
        assert torch.allclose(output1, output2)
    
    def test_model_parameters_count(self):
        """Test model has expected number of parameters."""
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\nActual parameter count: {total_params:,}")
        
        # Should be around 400K parameters (allow wider range)
        # Actual count may vary based on implementation details
        assert 200_000 < total_params < 600_000, f"Parameter count {total_params} is outside expected range"
    
    def test_model_output_range(self):
        """Test model output logits are in reasonable range."""
        model = SignLanguageTransformer(
            feature_dim=126,
            num_classes=100,
            d_model=192,
            num_layers=2,
            dropout=0.4
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 64, 126)
        
        # Forward pass
        output = model(x)
        
        # Logits should be in reasonable range (not NaN or Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() < 100  # Reasonable logit range
