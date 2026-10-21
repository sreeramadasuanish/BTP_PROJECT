"""Run complete end-to-end suite across all datasets and scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run_phase1 import run_phase1_pipeline
from src.pipeline.run_phase2 import run_phase2_pipeline
from src.utils.logger import get_logger

logger = get_logger("fed4fed.scripts.run_all")


def main():
    benchmarks = [
        ("MNIST", "Fair"),
        ("MNIST", "Unfair"),
        ("Fashion-MNIST", "Fair"),
        ("Fashion-MNIST", "Unfair"),
        ("CIFAR-10", "Fair"),
        ("CIFAR-10", "Unfair"),
        ("SST", "Fair"),
        ("SST", "Unfair"),
    ]

    for ds_name, sc_name in benchmarks:
        logger.info(f"========== Running Benchmark: {ds_name} - {sc_name} ==========")
        p1 = run_phase1_pipeline(
            dataset_name=ds_name,
            scenario=sc_name,
            num_clients=20 if ds_name == "SST" else 50,
            num_rounds=3,
            local_epochs=1,
            batch_size=32,
            lr=0.01,
            seed=42
        )

        exp_dir = p1["export_result"]["directory"]
        p2 = run_phase2_pipeline(
            phase1_exp_dir=exp_dir,
            num_bootstrap_resamples=100,
            significance_level=0.05,
            seed=42
        )

        logger.info(f"Verdict for {ds_name} [{sc_name}]: {p2['fairness_results']['overall_verdict']}")

    logger.info("All benchmarks executed successfully.")


if __name__ == "__main__":
    main()
