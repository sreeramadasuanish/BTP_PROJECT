"""Formatting numerical results into summary tables, CSV, and JSON."""

from typing import Any, Dict, List
import pandas as pd


def build_client_summary_table(client_statistics_df: pd.DataFrame) -> pd.DataFrame:
    """Format per-client bootstrap statistics for UI presentation."""
    df = client_statistics_df.copy()
    if "observed_metric" in df.columns:
        df["observed_metric"] = df["observed_metric"].map(lambda x: f"{x:.4f}")
    if "bootstrap_mean" in df.columns:
        df["bootstrap_mean"] = df["bootstrap_mean"].map(lambda x: f"{x:.4f}")
    if "bootstrap_variance" in df.columns:
        df["bootstrap_variance"] = df["bootstrap_variance"].map(lambda x: f"{x:.6f}")
    if "bootstrap_se" in df.columns:
        df["bootstrap_se"] = df["bootstrap_se"].map(lambda x: f"{x:.4f}")
    return df


def build_fairness_summary_table(fairness_results: Dict[str, Any]) -> pd.DataFrame:
    """Create comparison table between Algorithm 1 and Algorithm 2."""
    rows = []
    
    # Alg 1
    if fairness_results.get("alg1_result"):
        a1 = fairness_results["alg1_result"]
        rows.append({
            "Algorithm": "Algorithm 1 (Graybill-Deal / T1)",
            "Test Statistic": f"{a1['test_statistic']:.4f}",
            "Degrees of Freedom": str(a1["degrees_of_freedom"]),
            "p-value": f"{a1['p_value']:.4e}" if a1['p_value'] < 1e-3 else f"{a1['p_value']:.4f}",
            "Significance Level (α)": f"{a1['significance_level']:.3f}",
            "Decision": "Reject H0 (Disparity)" if a1["reject_h0"] else "Fail to Reject H0 (Fair)",
        })

    # Alg 2
    if fairness_results.get("alg2_result"):
        a2 = fairness_results["alg2_result"]
        rows.append({
            "Algorithm": "Algorithm 2 (Welch-Type F)",
            "Test Statistic": f"{a2['test_statistic']:.4f}",
            "Degrees of Freedom": f"({a2['df_numerator']:.0f}, {a2['df_denominator']:.1f})",
            "p-value": f"{a2['p_value']:.4e}" if a2['p_value'] < 1e-3 else f"{a2['p_value']:.4f}",
            "Significance Level (α)": f"{a2['significance_level']:.3f}",
            "Decision": "Reject H0 (Disparity)" if a2["reject_h0"] else "Fail to Reject H0 (Fair)",
        })

    return pd.DataFrame(rows)
