"""CLI script to run Phase 1 Federated Learning experiment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run_phase1 import run_phase1_pipeline
from src.utils.logger import get_logger

logger = get_logger("fed4fed.scripts.run_phase1")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Federated Learning.")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "Fashion-MNIST", "CIFAR-10", "SST"])
    parser.add_argument("--scenario", type=str, default="Fair", choices=["Fair", "Unfair"])
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha for CIFAR-10 unfair")
    args = parser.parse_args()

    p_kwargs = {}
    if args.dataset == "CIFAR-10" and args.scenario == "Unfair":
        p_kwargs["alpha"] = args.alpha

    res = run_phase1_pipeline(
        dataset_name=args.dataset,
        scenario=args.scenario,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer_type=args.optimizer,
        seed=args.seed,
        device=args.device,
        partition_kwargs=p_kwargs
    )

    logger.info(f"Phase 1 experiment completed: {res['experiment_id']}")
    logger.info(f"Export directory: {res['export_result']['directory']}")


if __name__ == "__main__":
    main()
