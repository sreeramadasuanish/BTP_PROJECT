"""Fairness evaluation engine orchestrating Algorithm 1, Algorithm 2, and decision policies."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.phase2_fed4fed.alg1_graybill_deal import run_algorithm_1
from src.phase2_fed4fed.alg2_welch import run_algorithm_2
from src.phase2_fed4fed.alg3_confidence import run_algorithm_3


def evaluate_fairness(
    client_means: np.ndarray,
    client_variances: np.ndarray,
    client_dfs: Optional[np.ndarray] = None,
    run_alg1: bool = True,
    run_alg2: bool = True,
    significance_level: float = 0.05,
    confidence_level: float = 0.95,
    decision_policy: str = "conservative"  # 'conservative' (reject if either rejects), 'unanimous' (reject if both reject)
) -> Dict[str, Any]:
    """Orchestrate complete Fed4Fed fairness evaluation across selected algorithms."""
    results: Dict[str, Any] = {
        "significance_level": significance_level,
        "confidence_level": confidence_level,
        "decision_policy": decision_policy,
        "algorithms_run": [],
        "alg1_result": None,
        "alg2_result": None,
        "overall_verdict": None,
        "reject_h0": False,
        "confidence_interval": None,
    }

    # Run Algorithm 1
    if run_alg1:
        res1 = run_algorithm_1(client_means, client_variances, significance_level=significance_level)
        results["alg1_result"] = res1
        results["algorithms_run"].append("Algorithm 1 (Graybill-Deal / Chi-Square T1)")

    # Run Algorithm 2
    if run_alg2:
        res2 = run_algorithm_2(client_means, client_variances, client_dfs=client_dfs, significance_level=significance_level)
        results["alg2_result"] = res2
        results["algorithms_run"].append("Algorithm 2 (Welch-Type F Test)")

    # Determine overall fairness decision based on selected algorithms and policy
    if run_alg1 and run_alg2:
        rej1 = results["alg1_result"]["reject_h0"]
        rej2 = results["alg2_result"]["reject_h0"]

        if decision_policy == "conservative":
            # Reject if either test finds evidence of disparity (conservative against unfairness)
            overall_reject = rej1 or rej2
            policy_desc = "Conservative Joint Policy (Disparity detected if either Alg 1 or Alg 2 rejects H0)"
        else:
            # Unanimous agreement required
            overall_reject = rej1 and rej2
            policy_desc = "Unanimous Policy (Disparity declared only if both Alg 1 and Alg 2 reject H0)"

        results["policy_description"] = policy_desc
        results["reject_h0"] = overall_reject

        if overall_reject:
            results["overall_verdict"] = "UNFAIR (Statistical Disparity Detected)"
            results["plain_english"] = (
                f"Under the {policy_desc}, the global federated model exhibits statistically significant "
                f"performance disparities across clients at alpha = {significance_level}. "
                f"Alg 1 p-value: {results['alg1_result']['p_value']:.4e}, "
                f"Alg 2 p-value: {results['alg2_result']['p_value']:.4e}."
            )
        else:
            results["overall_verdict"] = "FAIR (Performance Homogeneous)"
            results["plain_english"] = (
                f"Under the {policy_desc}, the global federated model demonstrates statistically fair performance. "
                f"Neither test found significant evidence to reject the null hypothesis of equal performance at alpha = {significance_level}. "
                f"Alg 1 p-value: {results['alg1_result']['p_value']:.4f}, "
                f"Alg 2 p-value: {results['alg2_result']['p_value']:.4f}."
            )

    elif run_alg1:
        rej1 = results["alg1_result"]["reject_h0"]
        results["reject_h0"] = rej1
        results["overall_verdict"] = "UNFAIR (Statistical Disparity Detected)" if rej1 else "FAIR (Performance Homogeneous)"
        results["plain_english"] = results["alg1_result"]["interpretation"]

    elif run_alg2:
        rej2 = results["alg2_result"]["reject_h0"]
        results["reject_h0"] = rej2
        results["overall_verdict"] = "UNFAIR (Statistical Disparity Detected)" if rej2 else "FAIR (Performance Homogeneous)"
        results["plain_english"] = results["alg2_result"]["interpretation"]

    else:
        raise ValueError("At least one statistical algorithm (Alg 1 or Alg 2) must be selected.")

    # Primary algorithm result to govern Algorithm 3 condition
    primary_test = results["alg2_result"] if run_alg2 else results["alg1_result"]
    primary_test["reject_h0"] = results["reject_h0"]  # Inherit overall verdict

    # Run Algorithm 3
    ci_res = run_algorithm_3(
        client_means=client_means,
        client_variances=client_variances,
        client_dfs=client_dfs,
        fairness_test_result=primary_test,
        confidence_level=confidence_level
    )
    results["confidence_interval"] = ci_res

    return results
