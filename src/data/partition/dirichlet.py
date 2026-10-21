"""Dirichlet label distribution partitioning for CIFAR-10."""

from typing import Dict, List
import numpy as np


def partition_dirichlet(
    targets: np.ndarray | List[int],
    num_clients: int,
    alpha: float = 0.1,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Partition dataset using Dirichlet distribution over class proportions.
    
    alpha controls heterogeneity:
    - alpha -> 0 (e.g. 0.001, 0.01): extreme non-IID, each client holds very few classes.
    - alpha = 0.1: moderate non-IID.
    - alpha = 10.0: nearly IID / uniform distribution.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive")

    targets = np.array(targets)
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    rng = np.random.RandomState(seed)

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in unique_classes:
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        n_c = len(idx_c)

        # Draw proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        # Scale proportions to sample counts
        counts = (proportions * n_c).astype(int)
        
        # Ensure sum matches n_c
        diff = n_c - counts.sum()
        for i in range(diff):
            counts[i % num_clients] += 1

        # Allocate slices to clients
        curr = 0
        for cid in range(num_clients):
            cnt = counts[cid]
            if cnt > 0:
                client_indices[cid].extend(idx_c[curr:curr + cnt].tolist())
                curr += cnt

    # Ensure every client has at least 1 sample if total samples > num_clients
    empty_clients = [cid for cid, inds in client_indices.items() if len(inds) == 0]
    for ec in empty_clients:
        # Steal one sample from the client with most samples
        max_client = max(client_indices.keys(), key=lambda k: len(client_indices[k]))
        if len(client_indices[max_client]) > 2:
            stolen = client_indices[max_client].pop()
            client_indices[ec].append(stolen)

    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    return client_indices
