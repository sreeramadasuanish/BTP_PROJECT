"""Unit tests for dataset split counts, shapes, and vocabulary."""

import pytest
import torch
from src.data.registry import load_dataset, get_dataset_metadata
from src.data.sst import load_sst


def test_mnist_dimensions():
    train_ds, test_ds, meta = load_dataset("MNIST")
    assert len(train_ds) == 60000
    assert len(test_ds) == 10000
    x, y = train_ds[0]
    assert x.shape == (1, 28, 28)
    assert 0 <= y.item() <= 9


def test_fashion_mnist_dimensions():
    train_ds, test_ds, meta = load_dataset("Fashion-MNIST")
    assert len(train_ds) == 60000
    assert len(test_ds) == 10000
    x, y = train_ds[0]
    assert x.shape == (1, 28, 28)


def test_cifar10_dimensions():
    train_ds, test_ds, meta = load_dataset("CIFAR-10")
    assert len(train_ds) == 50000
    assert len(test_ds) == 10000
    x, y = train_ds[0]
    assert x.shape == (3, 32, 32)


def test_sst_exact_split_counts():
    train_ds, val_ds, test_ds, vocab = load_sst()
    assert len(train_ds) == 8544, f"Expected 8544 train sentences, got {len(train_ds)}"
    assert len(val_ds) == 1101, f"Expected 1101 val sentences, got {len(val_ds)}"
    assert len(test_ds) == 2210, f"Expected 2210 test sentences, got {len(test_ds)}"
    x, y = train_ds[0]
    assert x.shape == (64,)
    assert y.item() in (0, 1)
    assert len(vocab) > 10
