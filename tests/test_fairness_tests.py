"""Unit tests for statistical fairness algorithms against independently computed examples."""

import numpy as np
import pytest
from scipy import stats

from src.phase2_fed4fed.alg1_graybill_deal import run_algorithm_1
from src.phase2_fed4fed.alg2_welch import run_algorithm_2
from src.phase2_fed4fed.alg3_confidence import run_algorithm_3
from src.phase2_fed4fed.fairness import evaluate_fairness


def test_alg1_exact_manual_calculation():
    # 2 clients with identical means -> T1 must be exactly 0, p-value = 1.0
    means = np.array([0.80, 0.80])
    variances = np.array([0.01, 0.01])

    res = run_algorithm_1(means, variances, significance_level=0.05)
    assert res["degrees_of_freedom"] == 1
    assert np.isclose(res["test_statistic"], 0.0)
    assert np.isclose(res["p_value"], 1.0)
    assert res["reject_h0"] is False


def test_alg1_known_disparity_rejection():
    # 2 clients with massive gap: 0.95 vs 0.50, low variances
    means = np.array([0.95, 0.50])
    variances = np.array([0.0001, 0.0001])

    res = run_algorithm_1(means, variances, significance_level=0.05)
    assert res["test_statistic"] > 500.0
    assert res["p_value"] < 1e-10
    assert res["reject_h0"] is True


def test_alg2_welch_heteroscedasticity():
    # 3 clients with unequal variances: var = 0.001, 0.01, 0.05
    means = np.array([0.85, 0.85, 0.85])
    variances = np.array([0.001, 0.01, 0.05])
    dfs = np.array([100.0, 100.0, 100.0])

    res = run_algorithm_2(means, variances, client_dfs=dfs, significance_level=0.05)
    assert np.isclose(res["test_statistic"], 0.0)
    assert np.isclose(res["p_value"], 1.0)
    assert res["reject_h0"] is False


def test_alg3_decision_rule_enforcement():
    # Case A: Fairness test accepts H0 -> Algorithm 3 permitted
    fair_result = {"reject_h0": False, "p_value": 0.5, "significance_level": 0.05}
    means = np.array([0.82, 0.84, 0.83])
    variances = np.array([0.001, 0.001, 0.001])

    ci_fair = run_algorithm_3(means, variances, fairness_test_result=fair_result)
    assert ci_fair["is_permitted"] is True
    assert ci_fair["status"] == "COMPUTED"
    assert ci_fair["pooled_estimate"] is not None
    assert ci_fair["pooled_ci_lower"] < ci_fair["pooled_estimate"] < ci_fair["pooled_ci_upper"]

    # Case B: Fairness test rejects H0 -> Algorithm 3 UNAVAILABLE under decision rule
    unfair_result = {"reject_h0": True, "p_value": 0.0001, "significance_level": 0.05, "algorithm": "Algorithm 1"}
    ci_unfair = run_algorithm_3(means, variances, fairness_test_result=unfair_result)
    assert ci_unfair["is_permitted"] is False
    assert ci_unfair["status"] == "UNAVAILABLE"
    assert "rejected the null hypothesis" in ci_unfair["reason"]


def test_significance_threshold_boundary():
    means = np.array([0.80, 0.83])
    variances = np.array([0.01, 0.01])
    res = run_algorithm_1(means, variances, significance_level=0.05)
    # Check decision changes when threshold is shifted across p_value
    p = res["p_value"]
    res_high_alpha = run_algorithm_1(means, variances, significance_level=p + 0.01)
    res_low_alpha = run_algorithm_1(means, variances, significance_level=p - 0.01)

    assert res_high_alpha["reject_h0"] is True
    assert res_low_alpha["reject_h0"] is False
