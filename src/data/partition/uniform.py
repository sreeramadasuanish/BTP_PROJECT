"""Uniform random IID data partitioning across clients."""

from typing import Dict, List
import numpy as np


def partition_uniform(
    num_samples: int,
    num_clients: int,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Partition indices uniformly and randomly among clients (IID).
    
    Guarantees no duplicated indices and complete coverage of sample space.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if num_samples < num_clients:
        raise ValueError(f"num_samples ({num_samples}) must be >= num_clients ({num_clients})")

    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    # Split into num_clients chunks
    client_indices: Dict[int, List[int]] = {}
    splits = np.array_split(indices, num_clients)

    for cid, split in enumerate(splits):
        client_indices[cid] = split.tolist()

    return client_indices
