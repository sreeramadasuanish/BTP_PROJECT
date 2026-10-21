"""Unit tests for bootstrap resampling, reproducibility, and metrics."""

import numpy as np
import pytest
from src.phase2_fed4fed.bootstrap import bootstrap_client_metric, run_clients_bootstrap


def test_bootstrap_reproducibility():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

    res1 = bootstrap_client_metric(y_true, y_pred, num_resamples=100, seed=42)
    res2 = bootstrap_client_metric(y_true, y_pred, num_resamples=100, seed=42)

    assert np.allclose(res1["distribution"], res2["distribution"])
    assert res1["bootstrap_mean"] == res2["bootstrap_mean"]
    assert res1["bootstrap_variance"] == res2["bootstrap_variance"]


def test_bootstrap_accuracy_and_moments():
    # 8 correct out of 10 -> accuracy 0.8
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    y_pred[8:] = 1  # 2 errors

    res = bootstrap_client_metric(y_true, y_pred, num_resamples=500, seed=123)
    assert res["observed"] == 0.8
    # Bootstrap mean should be closely aligned with observed point estimate
    assert abs(res["bootstrap_mean"] - 0.8) < 0.05
    assert res["bootstrap_variance"] > 0.0
    assert res["bootstrap_se"] > 0.0


def test_bootstrap_empty_set():
    res = bootstrap_client_metric(np.array([]), np.array([]), num_resamples=50)
    assert res["sample_count"] == 0
    assert res["observed"] == 0.0
