"""Federated Learning client representation and local training logic."""

from typing import Any, Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data.preprocessing import TensorDatasetWrapper
from src.metrics.metrics import evaluate_model


class FederatedClient:
    """Represents a decentralized client participating in federated training."""

    def __init__(
        self,
        client_id: int,
        dataset: TensorDatasetWrapper,
        train_indices: List[int],
        test_indices: List[int],
        model_fn: Any,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.device = device
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.train_dataset = Subset(dataset, train_indices)
        self.test_dataset = Subset(dataset, test_indices)

        self.num_train_samples = len(train_indices)
        self.num_test_samples = len(test_indices)

        self.model = model_fn().to(device)

    def train_one_round(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01,
        optimizer_type: str = "SGD"
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """Perform local training given latest global weights."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
        self.model.to(self.device)
        self.model.train()

        if self.num_train_samples == 0:
            return self.model.state_dict(), 0, 0.0, 0.0

        # Optimized loader parameters to prevent CPU training bottlenecks
        loader = DataLoader(
            self.train_dataset,
            batch_size=min(batch_size, self.num_train_samples),
            shuffle=True,
            num_workers=2 if self.device != "cpu" else 0,
            pin_memory=True if self.device != "cpu" else False
        )

        criterion = nn.CrossEntropyLoss()
        if optimizer_type.upper() == "ADAM":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        avg_loss = total_loss / total if total > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0.0

        cpu_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return cpu_state_dict, self.num_train_samples, avg_loss, avg_acc

    def evaluate_local_test(
        self,
        state_dict: Dict[str, torch.Tensor] | None = None
    ) -> Dict[str, float]:
        """Evaluate performance on client's private test partition."""
        if state_dict is not None:
            self.model.load_state_dict(copy.deepcopy(state_dict))
        self.model.to(self.device)

        if self.num_test_samples == 0:
            return {"accuracy": 0.0, "loss": 0.0, "sample_count": 0}

        loader = DataLoader(self.test_dataset, batch_size=min(64, self.num_test_samples), shuffle=False)
        return evaluate_model(self.model, loader, device=self.device)

    def get_test_predictions_and_targets(
        self,
        model: nn.Module
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Obtain individual sample-level predictions and ground truth."""
        model.eval()
        model.to(self.device)

        if self.num_test_samples == 0:
            return np.array([]), np.array([])

        loader = DataLoader(self.test_dataset, batch_size=min(64, self.num_test_samples), shuffle=False)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                out = model(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        return np.array(all_preds), np.array(all_targets)