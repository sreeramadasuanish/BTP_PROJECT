
"""Load the genuine CIFAR-10 dataset with caching."""

from pathlib import Path
from typing import Tuple

import torch
import torchvision

from src.data.preprocessing import TensorDatasetWrapper
from src.utils.config import get_base_dir
from src.utils.logger import get_logger

logger = get_logger("fed4fed.data.cifar10")


def load_cifar10(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Tuple[TensorDatasetWrapper, TensorDatasetWrapper]:
    """Load genuine CIFAR-10: 50,000 train and 10,000 test images."""

    base_dir = get_base_dir()

    raw_dir = Path(raw_dir or base_dir / "data" / "raw" / "cifar10")
    processed_dir = Path(
        processed_dir or base_dir / "data" / "processed" / "cifar10"
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_file = processed_dir / "cifar10_cache.pt"
    source_marker = "torchvision_cifar10_real_v1"

    # Load only caches explicitly written by this real-data loader.
    if cache_file.exists():
        try:
            cached = torch.load(cache_file, map_location="cpu", weights_only=True)

            if cached.get("source") == source_marker:
                train_data = cached["train_data"]
                train_targets = cached["train_targets"]
                test_data = cached["test_data"]
                test_targets = cached["test_targets"]

                if (
                    train_data.shape == (50000, 3, 32, 32)
                    and test_data.shape == (10000, 3, 32, 32)
                    and train_targets.shape == (50000,)
                    and test_targets.shape == (10000,)
                    and train_targets.min().item() >= 0
                    and train_targets.max().item() <= 9
                    and test_targets.min().item() >= 0
                    and test_targets.max().item() <= 9
                ):
                    logger.info("Loaded verified-format CIFAR-10 cache.")
                    return (
                        TensorDatasetWrapper(train_data, train_targets),
                        TensorDatasetWrapper(test_data, test_targets),
                    )

            logger.warning(
                "Cache is untrusted or invalid. Reloading official CIFAR-10."
            )

        except Exception as exc:
            logger.warning(f"Could not read cache: {exc}. Reloading dataset.")

    # download=True downloads official files when they are missing.
    # No synthetic fallback is permitted.
    try:
        logger.info("Loading official CIFAR-10 through torchvision...")

        train_ds = torchvision.datasets.CIFAR10(
            root=str(raw_dir),
            train=True,
            download=True,
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=str(raw_dir),
            train=False,
            download=True,
        )

        # Official CIFAR-10 images: (N, 32, 32, 3), uint8.
        # Convert to (N, 3, 32, 32), float32 in [0, 1].
        train_data = (
            torch.from_numpy(train_ds.data.copy())
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        test_data = (
            torch.from_numpy(test_ds.data.copy())
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )

        train_targets = torch.tensor(train_ds.targets, dtype=torch.long)
        test_targets = torch.tensor(test_ds.targets, dtype=torch.long)

        # Basic integrity checks.
        assert train_data.shape == (50000, 3, 32, 32)
        assert test_data.shape == (10000, 3, 32, 32)
        assert train_targets.shape == (50000,)
        assert test_targets.shape == (10000,)
        assert train_targets.min().item() >= 0
        assert train_targets.max().item() <= 9
        assert test_targets.min().item() >= 0
        assert test_targets.max().item() <= 9

    except Exception as exc:
        raise RuntimeError(
            "Could not load genuine CIFAR-10. Check your internet connection, "
            "torchvision installation, and dataset files. No synthetic data "
            "was generated. Fix the error and try again."
        ) from exc

    torch.save(
        {
            "source": source_marker,
            "train_data": train_data,
            "train_targets": train_targets,
            "test_data": test_data,
            "test_targets": test_targets,
        },
        cache_file,
    )

    logger.info(
        "Genuine CIFAR-10 loaded: %d training and %d test images.",
        len(train_targets),
        len(test_targets),
    )

    return (
        TensorDatasetWrapper(train_data, train_targets),
        TensorDatasetWrapper(test_data, test_targets),
    )
