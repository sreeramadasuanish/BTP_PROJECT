"""Unit tests for FedAvg mathematical weighting and parameter aggregation."""

from collections import OrderedDict
import pytest
import torch
from src.phase1_fl.fedavg import fedavg_aggregate


def test_fedavg_weighted_average_calculation():
    # Client 1: 100 samples, weight parameter = 2.0
    state1 = OrderedDict({
        "layer.weight": torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([2.0, 2.0], dtype=torch.float32),
    })

    # Client 2: 300 samples, weight parameter = 6.0
    state2 = OrderedDict({
        "layer.weight": torch.tensor([[6.0, 6.0], [6.0, 6.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([6.0, 6.0], dtype=torch.float32),
    })

    # Hand-calculated weighted average:
    # (100 * 2.0 + 300 * 6.0) / (100 + 300) = (200 + 1800) / 400 = 2000 / 400 = 5.0
    aggregated = fedavg_aggregate([(state1, 100), (state2, 300)])

    expected_w = torch.tensor([[5.0, 5.0], [5.0, 5.0]], dtype=torch.float32)
    expected_b = torch.tensor([5.0, 5.0], dtype=torch.float32)

    assert torch.allclose(aggregated["layer.weight"], expected_w)
    assert torch.allclose(aggregated["layer.bias"], expected_b)


def test_fedavg_empty_input_error():
    with pytest.raises(ValueError):
        fedavg_aggregate([])


def test_fedavg_zero_samples_error():
    state = OrderedDict({"w": torch.tensor([1.0])})
    with pytest.raises(ValueError):
        fedavg_aggregate([(state, 0)])
