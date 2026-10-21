"""Label-skew partitioning for binary text sentiment dataset (SST)."""

from typing import Dict, List
import numpy as np


def partition_label_skew(
    targets: np.ndarray | List[int],
    num_clients: int,
    skew_ratio: float = 0.90,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Partition binary dataset such that clients have heavy class imbalance."""
    targets = np.array(targets)
    rng = np.random.RandomState(seed)

    idx_pos = rng.permutation(np.where(targets == 1)[0]).tolist()
    idx_neg = rng.permutation(np.where(targets == 0)[0]).tolist()

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    half_clients = num_clients // 2
    # Group 1 (clients 0..half-1): predominantly negative
    # Group 2 (clients half..num_clients-1): predominantly positive

    total_samples = len(targets)
    target_per_client = total_samples // num_clients

    for cid in range(num_clients):
        is_neg_dominant = (cid < half_clients)
        if is_neg_dominant:
            n_dom = int(target_per_client * skew_ratio)
            n_min = target_per_client - n_dom
            dom_pool, min_pool = idx_neg, idx_pos
        else:
            n_dom = int(target_per_client * skew_ratio)
            n_min = target_per_client - n_dom
            dom_pool, min_pool = idx_pos, idx_neg

        take_dom = dom_pool[:min(n_dom, len(dom_pool))]
        del dom_pool[:len(take_dom)]
        take_min = min_pool[:min(n_min, len(min_pool))]
        del min_pool[:len(take_min)]

        client_indices[cid].extend(take_dom + take_min)

    # Distribute leftovers
    leftovers = idx_pos + idx_neg
    rng.shuffle(leftovers)
    for i, idx in enumerate(leftovers):
        client_indices[i % num_clients].append(idx)

    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    return client_indices
