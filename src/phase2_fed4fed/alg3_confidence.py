"""Algorithm 3: Performance Confidence Interval for Global Model under Fed4Fed Decision Rules."""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats


def run_algorithm_3(
    client_means: np.ndarray,
    client_variances: np.ndarray,
    client_dfs: Optional[np.ndarray] = None,
    fairness_test_result: Optional[Dict[str, Any]] = None,
    confidence_level: float = 0.95,
    force_run: bool = False
) -> Dict[str, Any]:
    """Execute Algorithm 3: Compute Performance Confidence Interval.
    
    Decision Rule:
        Algorithm 3 estimates the confidence interval for the common performance mean of the global model.
        It requires that the fairness test fails to reject H0 (p >= alpha).
        If H0 is rejected (p < alpha), the model is heterogeneous/unfair, violating the common mean assumption.
    """
    means = np.array(client_means, dtype=np.float64)
    variances = np.array(client_variances, dtype=np.float64)
    k = len(means)

    if client_dfs is None:
        dfs = np.full(k, 199.0, dtype=np.float64)
    else:
        dfs = np.maximum(np.array(client_dfs, dtype=np.float64), 2.0)

    # 1. Check Precondition Decision Rule
    is_permitted = True
    rejection_reason = None

    if fairness_test_result is not None:
        # Check reject_h0 flag
        if fairness_test_result.get("reject_h0", False):
            is_permitted = False
            p_val = fairness_test_result.get("p_value", 0.0)
            alpha = fairness_test_result.get("significance_level", 0.05)
            alg_name = fairness_test_result.get("algorithm", "Fairness test")
            rejection_reason = (
                f"{alg_name} rejected the null hypothesis of equal performance "
                f"(p = {p_val:.4e} < alpha = {alpha}). "
                f"Because performance disparity is statistically significant across clients, "
                f"the common mean assumption mu_1 = ... = mu_K does not hold. "
                f"A pooled global confidence interval cannot be constructed under Algorithm 3 decision rules."
            )

    # Calculate per-client intervals regardless for diagnostics
    gamma = 1.0 - confidence_level
    per_client_cis: List[Dict[str, Any]] = []

    for i in range(k):
        df_i = dfs[i]
        s_i = float(np.sqrt(max(variances[i], 1e-9)))
        t_crit = float(stats.t.ppf(1.0 - gamma / 2.0, df=df_i))
        c_mean = float(means[i])
        ci_lower = max(0.0, c_mean - t_crit * s_i)
        ci_upper = min(1.0, c_mean + t_crit * s_i)
        per_client_cis.append({
            "client_id": i,
            "mean": c_mean,
            "variance": float(variances[i]),
            "standard_error": s_i,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        })

    if not is_permitted and not force_run:
        return {
            "algorithm": "Algorithm 3 (Confidence Interval)",
            "is_permitted": False,
            "status": "UNAVAILABLE",
            "reason": rejection_reason,
            "confidence_level": confidence_level,
            "pooled_estimate": None,
            "pooled_ci_lower": None,
            "pooled_ci_upper": None,
            "per_client_intervals": per_client_cis,
            "interpretation": rejection_reason,
        }

    # 2. Compute Pooled Graybill-Deal Confidence Interval
    safe_variances = np.maximum(variances, 1e-9)
    weights = 1.0 / safe_variances
    total_w = float(np.sum(weights))
    mu_gd = float(np.sum(weights * means) / total_w)

    # Variance with Meier-type higher order correction
    w_ratio = weights / total_w
    correction = 1.0 + 4.0 * float(np.sum((1.0 / dfs) * w_ratio * (1.0 - w_ratio)))
    var_gd = (1.0 / total_w) * correction
    se_gd = float(np.sqrt(max(var_gd, 1e-12)))

    # Welch-Satterthwaite effective degrees of freedom
    lambda_eff = float(np.sum((1.0 / dfs) * ((1.0 - w_ratio) ** 2)))
    eff_df = float((k ** 2 - 1) / (3.0 * max(lambda_eff, 1e-9)))

    # t critical value
    t_critical = float(stats.t.ppf(1.0 - gamma / 2.0, df=max(eff_df, 5.0)))
    ci_lower = float(max(0.0, mu_gd - t_critical * se_gd))
    ci_upper = float(min(1.0, mu_gd + t_critical * se_gd))

    interpretation = (
        f"At {confidence_level*100:.1f}% confidence level, the common global model performance "
        f"is estimated at {mu_gd:.4f} with confidence interval [{ci_lower:.4f}, {ci_upper:.4f}] "
        f"(SE = {se_gd:.4f}, effective df = {eff_df:.1f})."
    )

    return {
        "algorithm": "Algorithm 3 (Confidence Interval)",
        "is_permitted": True,
        "status": "COMPUTED",
        "reason": None,
        "confidence_level": confidence_level,
        "pooled_estimate": mu_gd,
        "pooled_variance": var_gd,
        "pooled_se": se_gd,
        "effective_df": eff_df,
        "pooled_ci_lower": ci_lower,
        "pooled_ci_upper": ci_upper,
        "per_client_intervals": per_client_cis,
        "interpretation": interpretation,
    }
