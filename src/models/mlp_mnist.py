"""Multi-Layer Perceptron for MNIST as specified in Fed4Fed paper."""

import torch
import torch.nn as nn


class MLPMnist(nn.Module):
    """MLP architecture for MNIST: 784 -> 50 (ReLU) -> 10."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 50, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
