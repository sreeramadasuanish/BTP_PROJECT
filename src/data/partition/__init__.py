"""Unified partition dispatcher for FED4FED."""

from typing import Any, Dict, List, Tuple
import numpy as np

from src.data.partition.uniform import partition_uniform
from src.data.partition.clustered import partition_clustered
from src.data.partition.category import partition_category
from src.data.partition.dirichlet import partition_dirichlet
from src.data.partition.label_skew import partition_label_skew


def create_partitions(
    dataset_name: str,
    scenario: str,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    num_clients: int = 100,
    seed: int = 42,
    **kwargs: Any
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, Any]]:
    """Create train and test partitions according to dataset and scenario.
    
    Returns:
        (train_client_partitions, test_client_partitions, partition_metadata)
    """
    d_name = dataset_name.upper().replace("-", "_")
    is_fair = (scenario.lower() == "fair")

    train_n = len(train_targets)
    test_n = len(test_targets)

    meta: Dict[str, Any] = {
        "dataset": dataset_name,
        "scenario": scenario,
        "num_clients": num_clients,
        "seed": seed,
        "is_fair": is_fair,
    }

    if is_fair:
        meta["method"] = "uniform"
        meta["description"] = "Uniform random IID allocation across clients"
        train_parts = partition_uniform(train_n, num_clients, seed=seed)
        test_parts = partition_uniform(test_n, num_clients, seed=seed + 1)

    else:
        # Unfair scenario dataset-specific rules
        if d_name == "MNIST":
            meta["method"] = "clustered"
            shards = kwargs.get("shards_per_client", 2)
            meta["shards_per_client"] = shards
            meta["description"] = f"Clustered non-IID shards ({shards} shards per client)"
            train_parts = partition_clustered(train_targets, num_clients, shards_per_client=shards, seed=seed)
            test_parts = partition_clustered(test_targets, num_clients, shards_per_client=shards, seed=seed + 1)

        elif d_name == "FASHION_MNIST":
            meta["method"] = "category"
            skew_ratio = kwargs.get("skew_ratio", 0.85)
            prim_classes = kwargs.get("primary_classes_per_client", 2)
            meta["skew_ratio"] = skew_ratio
            meta["primary_classes_per_client"] = prim_classes
            meta["description"] = f"Category proportion skew ({skew_ratio*100:.0f}% primary categories)"
            train_parts = partition_category(train_targets, num_clients, primary_classes_per_client=prim_classes, skew_ratio=skew_ratio, seed=seed)
            test_parts = partition_category(test_targets, num_clients, primary_classes_per_client=prim_classes, skew_ratio=skew_ratio, seed=seed + 1)

        elif d_name in ("CIFAR10", "CIFAR_10"):
            meta["method"] = "dirichlet"
            alpha = float(kwargs.get("alpha", 0.1))
            meta["alpha"] = alpha
            meta["description"] = f"Dirichlet non-IID label distribution (alpha={alpha})"
            train_parts = partition_dirichlet(train_targets, num_clients, alpha=alpha, seed=seed)
            test_parts = partition_dirichlet(test_targets, num_clients, alpha=alpha, seed=seed + 1)

        elif d_name == "SST":
            meta["method"] = "label_skew"
            skew_ratio = kwargs.get("skew_ratio", 0.90)
            meta["skew_ratio"] = skew_ratio
            meta["description"] = f"Label-skewed binary distribution ({skew_ratio*100:.0f}% dominant class)"
            train_parts = partition_label_skew(train_targets, num_clients, skew_ratio=skew_ratio, seed=seed)
            test_parts = partition_label_skew(test_targets, num_clients, skew_ratio=skew_ratio, seed=seed + 1)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Compute summary stats on partition sizes
    train_sizes = [len(v) for v in train_parts.values()]
    test_sizes = [len(v) for v in test_parts.values()]

    meta["train_size_stats"] = {
        "min": int(np.min(train_sizes)),
        "max": int(np.max(train_sizes)),
        "mean": float(np.mean(train_sizes)),
        "std": float(np.std(train_sizes)),
        "total": int(np.sum(train_sizes)),
    }
    meta["test_size_stats"] = {
        "min": int(np.min(test_sizes)),
        "max": int(np.max(test_sizes)),
        "mean": float(np.mean(test_sizes)),
        "std": float(np.std(test_sizes)),
        "total": int(np.sum(test_sizes)),
    }

    return train_parts, test_parts, meta
