"""Algorithm 1: Graybill-Deal / Chi-Square T1 Test for Federated Learning Performance Fairness."""

from typing import Any, Dict
import numpy as np
from scipy import stats


def run_algorithm_1(
    client_means: np.ndarray,
    client_variances: np.ndarray,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """Execute Algorithm 1: Graybill-Deal / Chi-square T1 test.
    
    Hypotheses:
        H0: mu_1 = mu_2 = ... = mu_K = mu (Model performance is statistically fair across clients)
        H1: exists i != j such that mu_i != mu_j (Model performance disparity exists across clients)
        
    Formulas:
        w_i = 1 / S_i^2
        mu_GD = (sum w_i * X_bar_i) / (sum w_i)
        T1 = sum w_i * (X_bar_i - mu_GD)^2
        df = K - 1
        p_value = 1 - CDF_chi2(T1, df)
    """
    means = np.array(client_means, dtype=np.float64)
    variances = np.array(client_variances, dtype=np.float64)
    k = len(means)

    if k < 2:
        raise ValueError(f"Algorithm 1 requires at least 2 clients, got {k}")

    # Numerical protection for near-zero variances
    eps = 1e-9
    safe_variances = np.maximum(variances, eps)

    # Weights and Graybill-Deal pooled estimator
    weights = 1.0 / safe_variances
    sum_weights = np.sum(weights)
    mu_gd = float(np.sum(weights * means) / sum_weights)

    # Test statistic T1
    diffs = means - mu_gd
    t1_stat = float(np.sum(weights * (diffs ** 2)))
    df = k - 1

    # p-value from Chi-square distribution
    p_value = float(stats.chi2.sf(t1_stat, df=df))

    # Decision rule
    reject_h0 = bool(p_value < significance_level)
    verdict = "Unfair (Reject H0: Disparity Detected)" if reject_h0 else "Fair (Fail to Reject H0: Homogeneous)"

    # Interpretation
    if reject_h0:
        interpretation = (
            f"The test statistic T1 = {t1_stat:.4f} exceeds the critical value at alpha = {significance_level} "
            f"(p = {p_value:.4e} < {significance_level}). We reject the null hypothesis of equal client performance. "
            f"There is statistically significant evidence of cross-client performance disparity."
        )
    else:
        interpretation = (
            f"The test statistic T1 = {t1_stat:.4f} does not exceed the critical value at alpha = {significance_level} "
            f"(p = {p_value:.4f} >= {significance_level}). We fail to reject the null hypothesis. "
            f"Cross-client performance differences are consistent with random estimation variation."
        )

    # Diagnostic warnings
    var_ratio = float(np.max(safe_variances) / np.min(safe_variances))
    warnings = []
    if var_ratio > 50.0:
        warnings.append(
            f"High variance ratio ({var_ratio:.1f}x) observed across clients. "
            f"Algorithm 2 (Welch-type test) is recommended to account for extreme heteroscedasticity."
        )

    return {
        "algorithm": "Algorithm 1 (Graybill-Deal / Chi-Square T1)",
        "null_hypothesis": "H0: mu_1 = mu_2 = ... = mu_K = mu (Equal client performance)",
        "alternative_hypothesis": "H1: mu_i != mu_j for some i != j (Performance disparity exists)",
        "k_clients": k,
        "degrees_of_freedom": df,
        "test_statistic": t1_stat,
        "p_value": p_value,
        "significance_level": significance_level,
        "graybill_deal_mean": mu_gd,
        "reject_h0": reject_h0,
        "verdict": verdict,
        "interpretation": interpretation,
        "variance_ratio": var_ratio,
        "warnings": warnings,
    }
