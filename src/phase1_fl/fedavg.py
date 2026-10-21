"""Federated Averaging (FedAvg) aggregation algorithm."""

from collections import OrderedDict
from typing import Dict, List, Tuple
import torch


def fedavg_aggregate(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    """Aggregate local model state dictionaries weighted by sample count.
    
    Formula:
        w_global = sum_{k=1}^K (n_k / N_total) * w_k
        
    Args:
        client_updates: List of tuples (client_state_dict, num_samples)
        
    Returns:
        Aggregated global state_dict
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty list of client updates.")

    total_samples = sum(num_samples for _, num_samples in client_updates)
    if total_samples <= 0:
        raise ValueError("Total number of samples across participating clients must be positive.")

    # Initialize accumulator with zeros using shape and dtype of first client
    first_state, _ = client_updates[0]
    aggregated_state: Dict[str, torch.Tensor] = OrderedDict()

    for key, param in first_state.items():
        if param.is_floating_point():
            aggregated_state[key] = torch.zeros_like(param, dtype=torch.float32)
        else:
            # For integer buffers (e.g. num_batches_tracked in BatchNorm)
            aggregated_state[key] = param.clone()

    for state_dict, num_samples in client_updates:
        weight = float(num_samples) / float(total_samples)
        for key in aggregated_state:
            if aggregated_state[key].is_floating_point():
                aggregated_state[key] += state_dict[key].to(dtype=torch.float32) * weight
            else:
                aggregated_state[key] = state_dict[key]

    return aggregated_state
