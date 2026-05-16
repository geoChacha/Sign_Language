"""
Neural Network Model for ANN-Based PSL Alphabet Recognition System

This module defines the AlphabetClassifier, a feedforward neural network
for static hand pose classification.

Architecture:
    Input (42) → FC1 (128) → BatchNorm → ReLU → Dropout(0.3)
              → FC2 (64)  → BatchNorm → ReLU → Dropout(0.3)
              → FC3 (num_classes) → Logits

The number of output classes is determined dynamically from the dataset,
supporting all alphabet classes present in the dataset directory.

Design Rationale:
    - BatchNorm after each linear layer for training stability
    - ReLU activation (standard for feedforward networks)
    - Dropout for regularization (prevents overfitting)
    - No softmax in forward pass (CrossEntropyLoss handles it internally)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT


class AlphabetClassifier(nn.Module):
    """
    Feedforward neural network for PSL alphabet classification.

    Takes normalized 42-dimensional hand coordinates as input and outputs
    logits for all detected alphabet classes.

    The output layer size is set dynamically based on the number of classes
    found in the dataset, so no code changes are needed when new alphabet
    folders are added.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
        num_classes: int = 23,   # default overridden at runtime from dataset
        dropout: float = DROPOUT,
    ):
        """
        Initialize the AlphabetClassifier.

        Args:
            input_dim:    Input dimension (default: 42 = 21 landmarks × 2 coords)
            hidden_dim_1: First hidden layer size  (default: 128)
            hidden_dim_2: Second hidden layer size (default: 64)
            num_classes:  Number of output classes — set from dataset at runtime
            dropout:      Dropout probability (default: 0.3)
        """
        super(AlphabetClassifier, self).__init__()

        self.num_classes = num_classes

        # Layer 1: input → hidden_1
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: hidden_1 → hidden_2
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.dropout2 = nn.Dropout(dropout)

        # Output layer: hidden_2 → num_classes
        self.fc3 = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 42)

        Returns:
            Logits tensor of shape (batch_size, num_classes).
            No softmax applied — CrossEntropyLoss handles that.
        """
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities via softmax.

        Args:
            x: Input tensor of shape (batch_size, 42)

        Returns:
            Probability tensor of shape (batch_size, num_classes).
            Each row sums to 1.0.
        """
        return F.softmax(self.forward(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices (argmax of logits).

        Args:
            x: Input tensor of shape (batch_size, 42)

        Returns:
            Class-index tensor of shape (batch_size,)
        """
        return torch.argmax(self.forward(x), dim=1)
