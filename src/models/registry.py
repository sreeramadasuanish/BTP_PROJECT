
"""Model registry and factory for FED4FED architectures."""

from typing import Any, Dict, Tuple

import torch.nn as nn

from src.models.mlp_mnist import MLPMnist
from src.models.cnn import FedCNN, CIFAR10CNN
from src.models.lstm_attention import LSTMAttention


MODEL_REGISTRY = {
    "MLP_MNIST": MLPMnist,
    "CNN_FashionMNIST": FedCNN,
    "CNN_CIFAR10": CIFAR10CNN,
    "LSTM_Attention": LSTMAttention,
}


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable


def get_model(
    dataset_name: str,
    config: Dict[str, Any] | None = None,
) -> nn.Module:
    """Create the appropriate model for the selected dataset."""

    name = dataset_name.upper().replace("-", "_")
    config = config or {}

    if name == "MNIST":
        return MLPMnist(
            input_dim=config.get("input_dim", 784),
            hidden_dim=config.get("hidden_dim", 50),
            num_classes=config.get("output_dim", 10),
        )

    elif name == "FASHION_MNIST":
        return FedCNN(
            in_channels=config.get("channels", 1),
            num_classes=config.get("num_classes", 10),
            dropout=config.get("dropout", 0.2),
            input_size=(28, 28),
        )

    elif name in ("CIFAR10", "CIFAR_10"):
        return CIFAR10CNN(
            in_channels=config.get("channels", 3),
            num_classes=config.get("num_classes", 10),
            dropout=config.get("dropout", 0.3),
        )

    elif name == "SST":
        return LSTMAttention(
            vocab_size=config.get("vocab_size", 10000),
            embedding_dim=config.get("embedding_dim", 100),
            hidden_dim=config.get("hidden_dim", 64),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.2),
        )

    else:
        raise ValueError(
            f"Unknown dataset for model creation: {dataset_name}"
        )
