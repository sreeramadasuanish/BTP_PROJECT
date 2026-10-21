
"""Dataset registry providing a unified interface for all four datasets."""

from copy import deepcopy
from typing import Any, Dict, Tuple

from src.data.preprocessing import TensorDatasetWrapper
from src.data.mnist import load_mnist
from src.data.fashion_mnist import load_fashion_mnist
from src.data.cifar10 import load_cifar10
from src.data.sst import load_sst


DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "MNIST": {
        "name": "MNIST",
        "type": "image",
        "num_classes": 10,
        "input_shape": "(1, 28, 28)",
        "train_samples": 60000,
        "test_samples": 10000,
        "classes": [str(i) for i in range(10)],
        "model_architecture": "MLP (784 -> 50 -> 10)",
        "fair_partition": "Uniform random IID allocation",
        "unfair_partition": "Clustered label distribution",
    },
    "Fashion-MNIST": {
        "name": "Fashion-MNIST",
        "type": "image",
        "num_classes": 10,
        "input_shape": "(1, 28, 28)",
        "train_samples": 60000,
        "test_samples": 10000,
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "model_architecture": "CNN (3 conv layers + 2 FC, dropout 0.2)",
        "fair_partition": "Uniform random IID allocation",
        "unfair_partition": "Category proportion skew across clients",
    },
    "CIFAR-10": {
        "name": "CIFAR-10",
        "type": "image",
        "num_classes": 10,
        "input_shape": "(3, 32, 32)",
        "train_samples": 50000,
        "test_samples": 10000,
        "classes": [
            "Airplane", "Automobile", "Bird", "Cat", "Deer",
            "Dog", "Frog", "Horse", "Ship", "Truck",
        ],
        "model_architecture": "CNN (3 conv layers + 2 FC, dropout 0.2)",
        "fair_partition": "Uniform random IID allocation",
        "unfair_partition": (
            "Dirichlet label distribution "
            "(alpha in {0.001, 0.01, 0.1, 10})"
        ),
    },
    "SST": {
        "name": "SST",
        "type": "text",
        "num_classes": 2,
        "input_shape": "Sequence (max_len=64, vocab=10000)",
        "train_samples": 8544,
        "val_samples": 1101,
        "test_samples": 2210,
        "classes": ["negative", "positive"],
        "model_architecture": "LSTM + Attention (embed=100, hidden=64)",
        "fair_partition": "Uniform class proportions across clients",
        "unfair_partition": "Label-skewed client distributions",
        "split_note": (
            "Custom subsets from labeled SST-2 training data; "
            "not the official validation/test splits."
        ),
    },
}


def get_dataset_metadata(dataset_name: str) -> Dict[str, Any]:
    """Return a copy of metadata for a registered dataset."""

    normalized = dataset_name.strip().lower().replace("-", "_")

    for name, metadata in DATASET_METADATA.items():
        if name.lower().replace("-", "_") == normalized:
            return deepcopy(metadata)

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Available datasets: {list(DATASET_METADATA.keys())}"
    )


def load_dataset(
    dataset_name: str,
) -> Tuple[TensorDatasetWrapper, TensorDatasetWrapper, Dict[str, Any]]:
    """
    Load a registered dataset.

    Returns:
        train_ds, test_ds, metadata

    For SST, metadata also contains:
        val_dataset: validation dataset
        vocab: vocabulary used to encode text
    """

    meta = get_dataset_metadata(dataset_name)
    name = meta["name"]

    if name == "MNIST":
        train_ds, test_ds = load_mnist()

    elif name == "Fashion-MNIST":
        train_ds, test_ds = load_fashion_mnist()

    elif name == "CIFAR-10":
        train_ds, test_ds = load_cifar10()

    elif name == "SST":
        train_ds, val_ds, test_ds, vocab = load_sst()
        meta["val_dataset"] = val_ds
        meta["vocab"] = vocab

    else:
        raise ValueError(f"Dataset loader not implemented: {name}")

    # Validate loaded sample counts.
    if len(train_ds) != meta["train_samples"]:
        raise ValueError(
            f"{name}: expected {meta['train_samples']} training samples, "
            f"got {len(train_ds)}."
        )

    if len(test_ds) != meta["test_samples"]:
        raise ValueError(
            f"{name}: expected {meta['test_samples']} test samples, "
            f"got {len(test_ds)}."
        )

    if name == "SST":
        if len(meta["val_dataset"]) != meta["val_samples"]:
            raise ValueError(
                f"SST: expected {meta['val_samples']} validation samples, "
                f"got {len(meta['val_dataset'])}."
            )

    return train_ds, test_ds, meta
