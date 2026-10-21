"""Category proportion skew partitioning for Fashion-MNIST."""

from typing import Dict, List
import numpy as np


def partition_category(
    targets: np.ndarray | List[int],
    num_clients: int,
    primary_classes_per_client: int = 2,
    skew_ratio: float = 0.85,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Partition dataset such that each client is heavily skewed towards specific clothing categories."""
    targets = np.array(targets)
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    rng = np.random.RandomState(seed)

    # Group sample indices by class
    class_indices = {c: rng.permutation(np.where(targets == c)[0]).tolist() for c in unique_classes}

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # Assign primary classes to each client
    for cid in range(num_clients):
        # Pick primary classes for this client
        prim = rng.choice(unique_classes, size=min(primary_classes_per_client, num_classes), replace=False)

        # Allocate quota from primary classes
        for pc in prim:
            n_take = max(1, int(len(class_indices[pc]) / (num_clients * primary_classes_per_client / 2)))
            take = class_indices[pc][:n_take]
            client_indices[cid].extend(take)
            class_indices[pc] = class_indices[pc][n_take:]

    # Distribute remaining samples round-robin across clients
    remaining = []
    for c in unique_classes:
        remaining.extend(class_indices[c])
    rng.shuffle(remaining)

    for i, idx in enumerate(remaining):
        client_indices[i % num_clients].append(idx)

    # Shuffle each client's final pool
    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    return client_indices
