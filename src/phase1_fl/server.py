"""Central server for orchestrating federated learning aggregation and evaluation."""

from typing import Any, Dict, List, Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.preprocessing import TensorDatasetWrapper
from src.metrics.metrics import evaluate_model
from src.phase1_fl.fedavg import fedavg_aggregate


class FederatedServer:
    """Central server maintaining the global model and coordinating FedAvg aggregation."""

    def __init__(
        self,
        model_fn: Any,
        val_dataset: TensorDatasetWrapper | None = None,
        device: str = "cpu"
    ):
        self.device = device
        self.global_model: nn.Module = model_fn().to(device)
        self.val_dataset = val_dataset
        self.val_loader = (
            DataLoader(val_dataset, batch_size=128, shuffle=False)
            if val_dataset is not None
            else None
        )

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return a detached clone of current global model weights on CPU."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def aggregate_updates(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate participating client updates via weighted FedAvg."""
        aggregated = fedavg_aggregate(client_updates)
        self.global_model.load_state_dict(aggregated)
        return aggregated

    def evaluate_global(self) -> Dict[str, float]:
        """Evaluate global model on validation/test set."""
        if self.val_loader is None:
            return {"accuracy": 0.0, "loss": 0.0}
        return evaluate_model(self.global_model, self.val_loader, device=self.device)
