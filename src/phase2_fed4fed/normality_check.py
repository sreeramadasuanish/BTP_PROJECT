"""Diagnostic normality tests for bootstrap distributions."""

from typing import Any, Dict
import numpy as np
from scipy import stats


def evaluate_normality(distribution: np.ndarray) -> Dict[str, Any]:
    """Test whether empirical bootstrap distribution conforms to asymptotic normality."""
    dist = np.array(distribution).flatten()
    n = len(dist)
    if n < 8 or np.all(dist == dist[0]):
        return {
            "is_normal": True,
            "shapiro_p": 1.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "warning": "Constant or insufficient data for normality test."
        }

    # Shapiro-Wilk test
    if n <= 5000:
        stat, p_val = stats.shapiro(dist)
    else:
        stat, p_val = stats.normaltest(dist)

    skew = float(stats.skew(dist))
    kurt = float(stats.kurtosis(dist))

    # Q-Q plot data coordinates
    sorted_dist = np.sort(dist)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "is_normal": bool(p_val >= 0.05),
        "skewness": skew,
        "kurtosis": kurt,
        "theoretical_quantiles": theoretical_quantiles.tolist(),
        "empirical_quantiles": sorted_dist.tolist(),
    }
