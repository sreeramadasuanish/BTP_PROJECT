"""Unit tests for neural network model architectures and shapes."""

import torch
import pytest
from src.models.mlp_mnist import MLPMnist
from src.models.cnn import FedCNN
from src.models.lstm_attention import LSTMAttention
from src.models.registry import get_model, count_parameters


def test_mlp_mnist_forward():
    model = MLPMnist(input_dim=784, hidden_dim=50, num_classes=10)
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)


def test_cnn_fashion_mnist_forward():
    model = FedCNN(in_channels=1, num_classes=10, dropout=0.2, input_size=(28, 28))
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_cnn_cifar10_forward():
    model = FedCNN(in_channels=3, num_classes=10, dropout=0.2, input_size=(32, 32))
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_lstm_attention_forward():
    model = LSTMAttention(vocab_size=1000, embedding_dim=64, hidden_dim=32, num_classes=2)
    x = torch.randint(0, 1000, (6, 64))
    out = model(x)
    assert out.shape == (6, 2)


def test_model_registry_factory():
    for name in ["MNIST", "Fashion-MNIST", "CIFAR-10", "SST"]:
        model = get_model(name)
        tot, tr = count_parameters(model)
        assert tot > 0
        assert tr == tot
