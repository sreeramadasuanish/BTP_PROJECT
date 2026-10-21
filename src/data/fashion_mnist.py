
"""Fashion-MNIST loader using genuine data with caching."""

from pathlib import Path
from typing import Tuple

import torch
import torchvision

from src.data.preprocessing import TensorDatasetWrapper
from src.utils.config import get_base_dir
from src.utils.logger import get_logger

logger = get_logger("fed4fed.data.fashion_mnist")


def load_fashion_mnist(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Tuple[TensorDatasetWrapper, TensorDatasetWrapper]:
    """Load genuine Fashion-MNIST: 60,000 train and 10,000 test images."""

    base_dir = get_base_dir()
    raw_dir = Path(
        raw_dir or base_dir / "data" / "raw" / "fashion_mnist"
    )
    processed_dir = Path(
        processed_dir or base_dir / "data" / "processed" / "fashion_mnist"
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_file = processed_dir / "fashion_mnist_cache.pt"
    source_marker = "torchvision_fashion_mnist_real_v1"

    # Reuse only caches created by this real-data loader.
    if cache_file.exists():
        try:
            cached = torch.load(
                cache_file, map_location="cpu", weights_only=True
            )

            if cached.get("source") == source_marker:
                train_data = cached["train_data"]
                train_targets = cached["train_targets"]
                test_data = cached["test_data"]
                test_targets = cached["test_targets"]

                valid = (
                    train_data.shape == (60000, 1, 28, 28)
                    and test_data.shape == (10000, 1, 28, 28)
                    and train_targets.shape == (60000,)
                    and test_targets.shape == (10000,)
                    and train_targets.min().item() >= 0
                    and train_targets.max().item() <= 9
                    and test_targets.min().item() >= 0
                    and test_targets.max().item() <= 9
                )

                if valid:
                    logger.info("Loaded Fashion-MNIST cache.")
                    return (
                        TensorDatasetWrapper(train_data, train_targets),
                        TensorDatasetWrapper(test_data, test_targets),
                    )

            logger.warning("Cache is untrusted or invalid; reloading dataset.")

        except Exception as exc:
            logger.warning(f"Cache could not be loaded: {exc}")

    # Download genuine Fashion-MNIST if the files are missing.
    try:
        logger.info("Loading official Fashion-MNIST through torchvision...")

        train_ds = torchvision.datasets.FashionMNIST(
            root=str(raw_dir),
            train=True,
            download=True,
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root=str(raw_dir),
            train=False,
            download=True,
        )

        train_data = train_ds.data.unsqueeze(1).float() / 255.0
        train_targets = train_ds.targets.long()

        test_data = test_ds.data.unsqueeze(1).float() / 255.0
        test_targets = test_ds.targets.long()

        assert train_data.shape == (60000, 1, 28, 28)
        assert test_data.shape == (10000, 1, 28, 28)
        assert train_targets.shape == (60000,)
        assert test_targets.shape == (10000,)

        assert train_targets.min().item() >= 0
        assert train_targets.max().item() <= 9
        assert test_targets.min().item() >= 0
        assert test_targets.max().item() <= 9

    except Exception as exc:
        raise RuntimeError(
            "Could not load genuine Fashion-MNIST. Check your internet "
            "connection, torchvision installation, and dataset files. "
            "No synthetic data was generated."
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
        "Fashion-MNIST loaded: %d training and %d test images.",
        len(train_targets),
        len(test_targets),
    )

    return (
        TensorDatasetWrapper(train_data, train_targets),
        TensorDatasetWrapper(test_data, test_targets),
    )
