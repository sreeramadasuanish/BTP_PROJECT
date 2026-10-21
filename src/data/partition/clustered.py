"""Clustered label non-IID partitioning for MNIST."""

from typing import Dict, List
import numpy as np


def partition_clustered(
    targets: np.ndarray | List[int],
    num_clients: int,
    shards_per_client: int = 2,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Partition dataset into non-IID shards sorted by label and assign shards to clients.
    
    Each client receives shards belonging predominantly to specific label clusters.
    """
    targets = np.array(targets)
    num_samples = len(targets)
    total_shards = num_clients * shards_per_client
    shard_size = num_samples // total_shards

    rng = np.random.RandomState(seed)

    # Sort indices by target label
    sorted_indices = np.argsort(targets)

    # Slice into total_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(total_shards)]

    # Handle remaining indices if any
    rem = sorted_indices[total_shards * shard_size:]
    if len(rem) > 0:
        for i, idx in enumerate(rem):
            shards[i % total_shards] = np.append(shards[i % total_shards], idx)

    # Permute shard assignment to clients
    shard_order = rng.permutation(total_shards)

    client_indices: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        client_shards = shard_order[cid * shards_per_client:(cid + 1) * shards_per_client]
        client_inds = np.concatenate([shards[s] for s in client_shards])
        rng.shuffle(client_inds)
        client_indices[cid] = client_inds.tolist()

    return client_indices
