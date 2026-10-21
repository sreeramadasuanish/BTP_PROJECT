"""Algorithm 2: Welch-Type F Test for Performance Fairness under Heteroscedasticity."""

from typing import Any, Dict
import numpy as np
from scipy import stats


def run_algorithm_2(
    client_means: np.ndarray,
    client_variances: np.ndarray,
    client_dfs: np.ndarray | None = None,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """Execute Algorithm 2: Welch-type F test for equality of client performance.
    
    Robust against unequal client variances (heteroscedasticity).
    
    Hypotheses:
        H0: mu_1 = mu_2 = ... = mu_K = mu (Equal client performance)
        H1: exists i != j such that mu_i != mu_j (Performance disparity exists)
        
    Formulas:
        w_i = 1 / S_i^2
        W = sum w_i
        X_star = (sum w_i * X_bar_i) / W
        Lambda = sum_{i=1}^K (1 / df_i) * (1 - w_i / W)^2
        F_W = [ (1 / (K - 1)) * sum w_i * (X_bar_i - X_star)^2 ] / [ 1 + (2 * (K - 2) / (K^2 - 1)) * Lambda ]
        nu_1 = K - 1
        nu_2 = (K^2 - 1) / (3 * Lambda)
        p_value = 1 - CDF_F(F_W, nu_1, nu_2)
    """
    means = np.array(client_means, dtype=np.float64)
    variances = np.array(client_variances, dtype=np.float64)
    k = len(means)

    if k < 2:
        raise ValueError(f"Algorithm 2 requires at least 2 clients, got {k}")

    # Default degrees of freedom per client if not provided
    if client_dfs is None:
        # Default bootstrap replication df = 199 or 99
        dfs = np.full(k, 199.0, dtype=np.float64)
    else:
        dfs = np.maximum(np.array(client_dfs, dtype=np.float64), 2.0)

    # Protection against zero variance
    eps = 1e-9
    safe_variances = np.maximum(variances, eps)

    weights = 1.0 / safe_variances
    total_w = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / total_w)

    # Compute Lambda
    w_ratio = weights / total_w
    lambda_term = float(np.sum((1.0 / dfs) * ((1.0 - w_ratio) ** 2)))

    # Prevent division by zero if Lambda is minuscule
    lambda_term = max(lambda_term, 1e-12)

    # Numerator sum of squares divided by (K - 1)
    diffs = means - weighted_mean
    ss_num = float(np.sum(weights * (diffs ** 2)))
    num_term = ss_num / (k - 1)

    # Denominator correction factor
    den_corr = 1.0 + (2.0 * (k - 2) / (k ** 2 - 1)) * lambda_term
    f_stat = float(num_term / den_corr)

    # Degrees of freedom
    nu1 = float(k - 1)
    nu2 = float((k ** 2 - 1) / (3.0 * lambda_term))

    # p-value from F distribution
    p_value = float(stats.f.sf(f_stat, dfn=nu1, dfd=nu2))

    # Decision rule
    reject_h0 = bool(p_value < significance_level)
    verdict = "Unfair (Reject H0: Disparity Detected)" if reject_h0 else "Fair (Fail to Reject H0: Homogeneous)"

    if reject_h0:
        interpretation = (
            f"The Welch-type test statistic F = {f_stat:.4f} with df = ({nu1:.0f}, {nu2:.1f}) is significant "
            f"(p = {p_value:.4e} < {significance_level}). We reject H0. Even after accounting for unequal variances, "
            f"there is strong evidence that the model does not perform equally across clients."
        )
    else:
        interpretation = (
            f"The Welch-type test statistic F = {f_stat:.4f} with df = ({nu1:.0f}, {nu2:.1f}) is not significant "
            f"(p = {p_value:.4f} >= {significance_level}). We fail to reject H0. Model performance is statistically "
            f"homogeneous across clients under heteroscedastic conditions."
        )

    return {
        "algorithm": "Algorithm 2 (Welch-Type F Test)",
        "null_hypothesis": "H0: mu_1 = mu_2 = ... = mu_K = mu (Equal client performance)",
        "alternative_hypothesis": "H1: mu_i != mu_j for some i != j (Performance disparity exists)",
        "k_clients": k,
        "df_numerator": nu1,
        "df_denominator": nu2,
        "test_statistic": f_stat,
        "p_value": p_value,
        "significance_level": significance_level,
        "weighted_mean": weighted_mean,
        "reject_h0": reject_h0,
        "verdict": verdict,
        "interpretation": interpretation,
        "lambda_correction": lambda_term,
    }
