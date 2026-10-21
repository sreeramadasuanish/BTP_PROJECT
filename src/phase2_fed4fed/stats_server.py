"""Server-side statistical aggregation for Fed4Fed."""

from typing import Any, Dict, List
import numpy as np


class StatsServer:
    """Server-side orchestrator that receives client statistical payloads and aggregates them."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def collect_payloads(self, payloads: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract and structure client parameters from encrypted/summary payloads."""
        cids = []
        means = []
        variances = []
        counts = []

        for p in payloads:
            # Supports both direct payload and encrypted wrapper
            c_data = p.get("encrypted_payload", p)
            cids.append(c_data["client_id"])
            means.append(c_data["mean"])
            variances.append(c_data["variance"])
            counts.append(c_data["sample_count"])

        # Prevent division by zero with small variance floor
        var_arr = np.array(variances, dtype=np.float64)
        min_pos_var = np.min(var_arr[var_arr > 0]) if np.any(var_arr > 0) else 1e-6
        var_arr = np.maximum(var_arr, min_pos_var * 1e-4 + 1e-8)

        return {
            "client_ids": np.array(cids),
            "means": np.array(means, dtype=np.float64),
            "variances": var_arr,
            "sample_counts": np.array(counts, dtype=np.int64),
        }
