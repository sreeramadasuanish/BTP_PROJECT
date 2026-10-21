"""CLI script to download, verify, and cache all 4 benchmark datasets."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.registry import DATASET_METADATA, load_dataset
from src.utils.logger import get_logger

logger = get_logger("fed4fed.scripts.download_data")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare FED4FED datasets.")
    parser.add_argument("--dataset", type=str, default="ALL", choices=["ALL", "MNIST", "Fashion-MNIST", "CIFAR-10", "SST"])
    args = parser.parse_args()

    targets = list(DATASET_METADATA.keys()) if args.dataset == "ALL" else [args.dataset]

    logger.info(f"Preparing datasets: {targets}...")
    for ds_name in targets:
        logger.info(f"--- Loading and verifying {ds_name} ---")
        try:
            train_ds, test_ds, meta = load_dataset(ds_name)
            logger.info(f"SUCCESS: {ds_name} ready. Train: {len(train_ds)}, Test: {len(test_ds)}")
        except Exception as e:
            logger.error(f"FAILURE on {ds_name}: {e}")

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
