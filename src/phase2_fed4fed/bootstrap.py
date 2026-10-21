"""Non-parametric bootstrap resampling on decentralized client test sets."""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocessing import TensorDatasetWrapper
from src.metrics.metrics import compute_accuracy, compute_error_rate


def bootstrap_client_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_resamples: int = 200,
    metric_name: str = "accuracy",
    seed: int = 42
) -> Dict[str, Any]:
    """Perform non-parametric bootstrap resampling on a client's test observations.
    
    Args:
        y_true: True class labels of length n_i
        y_pred: Model predicted class labels of length n_i
        num_resamples: B (number of bootstrap replications)
        metric_name: 'accuracy' or 'error_rate'
        seed: Random seed for bootstrap reproducibility
        
    Returns:
        Dict containing observed point estimate, bootstrap distribution, mean, variance, and standard error.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {
            "sample_count": 0,
            "observed": 0.0,
            "bootstrap_mean": 0.0,
            "bootstrap_variance": 0.0,
            "bootstrap_se": 0.0,
            "distribution": np.zeros(num_resamples),
            "warning": "Client test set is empty."
        }

    # 1. Observed point estimate on original private test set
    if metric_name == "error_rate":
        observed_val = compute_error_rate(y_true, y_pred)
        metric_fn = compute_error_rate
    else:
        observed_val = compute_accuracy(y_true, y_pred)
        metric_fn = compute_accuracy

    # 2. Non-parametric bootstrap resampling with replacement
    rng = np.random.RandomState(seed)
    boot_estimates = np.empty(num_resamples, dtype=np.float64)

    for b in range(num_resamples):
        resample_indices = rng.randint(0, n_samples, size=n_samples)
        boot_estimates[b] = metric_fn(y_true[resample_indices], y_pred[resample_indices])

    # 3. Bootstrap moments
    boot_mean = float(np.mean(boot_estimates))
    # Unbiased sample variance of bootstrap distribution
    boot_var = float(np.var(boot_estimates, ddof=1)) if num_resamples > 1 else 0.0
    boot_se = float(np.sqrt(boot_var))

    # Diagnostic warning for small test partitions
    warning = None
    if n_samples < 10:
        warning = f"Small test set ({n_samples} samples). Bootstrap estimates may have limited support."

    return {
        "sample_count": n_samples,
        "observed": float(observed_val),
        "bootstrap_mean": boot_mean,
        "bootstrap_variance": boot_var,
        "bootstrap_se": boot_se,
        "distribution": boot_estimates,
        "warning": warning,
    }


def run_clients_bootstrap(
    client_test_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    num_resamples: int = 200,
    metric_name: str = "accuracy",
    seed: int = 42
) -> Dict[str, Any]:
    """Execute bootstrap evaluation across all decentralized clients."""
    client_results: Dict[int, Dict[str, Any]] = {}
    bootstrap_matrix = []
    observed_points = []
    means = []
    variances = []
    sample_sizes = []

    for cid in sorted(client_test_data.keys()):
        y_true, y_pred = client_test_data[cid]
        c_seed = seed + cid
        res = bootstrap_client_metric(
            y_true=y_true,
            y_pred=y_pred,
            num_resamples=num_resamples,
            metric_name=metric_name,
            seed=c_seed
        )
        client_results[cid] = res
        bootstrap_matrix.append(res["distribution"])
        observed_points.append(res["observed"])
        means.append(res["bootstrap_mean"])
        variances.append(res["bootstrap_variance"])
        sample_sizes.append(res["sample_count"])

    return {
        "num_clients": len(client_results),
        "num_resamples": num_resamples,
        "metric_name": metric_name,
        "seed": seed,
        "clients": client_results,
        "bootstrap_data": np.array(bootstrap_matrix),  # shape: [K, B]
        "observed_points": np.array(observed_points),
        "bootstrap_means": np.array(means),
        "bootstrap_variances": np.array(variances),
        "sample_sizes": np.array(sample_sizes),
    }
