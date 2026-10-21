"""Sweep Dirichlet alpha parameter on CIFAR-10 to evaluate fairness sensitivity."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run_phase1 import run_phase1_pipeline
from src.pipeline.run_phase2 import run_phase2_pipeline
from src.utils.logger import get_logger

logger = get_logger("fed4fed.scripts.run_alpha_sweep")


def main():
    alphas = [0.001, 0.01, 0.1, 10.0]
    records = []

    for alpha in alphas:
        logger.info(f"--- Running CIFAR-10 Dirichlet alpha={alpha} ---")
        p1 = run_phase1_pipeline(
            dataset_name="CIFAR-10",
            scenario="Unfair",
            num_clients=20,
            num_rounds=3,
            partition_kwargs={"alpha": alpha},
            seed=42
        )

        p2 = run_phase2_pipeline(
            phase1_exp_dir=p1["export_result"]["directory"],
            num_bootstrap_resamples=100,
            significance_level=0.05,
            seed=42
        )

        f_res = p2["fairness_results"]
        records.append({
            "alpha": alpha,
            "overall_verdict": f_res["overall_verdict"],
            "alg1_pvalue": f_res["alg1_result"]["p_value"] if f_res.get("alg1_result") else None,
            "alg2_pvalue": f_res["alg2_result"]["p_value"] if f_res.get("alg2_result") else None,
            "experiment_id": p1["experiment_id"]
        })

    df = pd.DataFrame(records)
    print("\nAlpha Sweep Results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
