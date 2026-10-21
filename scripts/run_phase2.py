"""CLI script to run Phase 2 FED4FED statistical fairness evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run_phase2 import run_phase2_pipeline
from src.utils.logger import get_logger

logger = get_logger("fed4fed.scripts.run_phase2")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 FED4FED Statistical Evaluation.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to Phase 1 experiment export directory.")
    parser.add_argument("--resamples", type=int, default=200, help="Bootstrap replications B.")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "error_rate"])
    parser.add_argument("--alpha", type=float, default=0.05, help="Statistical significance level alpha.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (e.g. 0.95).")
    parser.add_argument("--policy", type=str, default="conservative", choices=["conservative", "unanimous"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    res = run_phase2_pipeline(
        phase1_exp_dir=args.exp_dir,
        num_bootstrap_resamples=args.resamples,
        metric_name=args.metric,
        significance_level=args.alpha,
        confidence_level=args.confidence,
        decision_policy=args.policy,
        seed=args.seed,
        device=args.device
    )

    logger.info("Phase 2 evaluation completed!")
    logger.info(f"Verdict: {res['fairness_results']['overall_verdict']}")
    logger.info(f"Report: {res['pdf_path']}")


if __name__ == "__main__":
    main()
