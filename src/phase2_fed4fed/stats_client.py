"""Client-side statistical computations for Fed4Fed."""

from typing import Any, Dict
import numpy as np

from src.phase2_fed4fed.bootstrap import bootstrap_client_metric
from src.phase2_fed4fed.encryption import SimulatedHomomorphicEncryption


class StatsClient:
    """Client-side statistical processor for Fed4Fed privacy-preserving evaluation."""

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.crypto = SimulatedHomomorphicEncryption()

    def process_test_data(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_resamples: int = 200,
        metric: str = "accuracy",
        seed: int = 42
    ) -> Dict[str, Any]:
        """Perform local bootstrap and return encrypted summary statistics."""
        boot_res = bootstrap_client_metric(
            y_true=y_true,
            y_pred=y_pred,
            num_resamples=num_resamples,
            metric_name=metric,
            seed=seed + self.client_id
        )

        encrypted_payload = self.crypto.encrypt_client_payload(
            client_id=self.client_id,
            mean=boot_res["bootstrap_mean"],
            variance=boot_res["bootstrap_variance"],
            n_samples=boot_res["sample_count"]
        )

        return {
            "client_id": self.client_id,
            "local_results": boot_res,
            "encrypted_payload": encrypted_payload
        }
