"""Simulation engine for Federated Learning training rounds."""

import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import TensorDatasetWrapper
from src.models.registry import get_model
from src.phase1_fl.client import FederatedClient
from src.phase1_fl.server import FederatedServer
from src.utils.logger import get_logger

logger = get_logger("fed4fed.phase1.simulation")


class FLSimulation:
    """Simulates communication rounds of Federated Learning across local clients."""

    def __init__(
        self,
        dataset_name: str,
        train_dataset: TensorDatasetWrapper,
        test_dataset: TensorDatasetWrapper,
        train_partitions: Dict[int, List[int]],
        test_partitions: Dict[int, List[int]],
        model_config: Dict[str, Any] | None = None,
        device: str = "cpu",
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.seed = seed
        self.model_config = model_config or {}

        def model_fn():
            return get_model(self.dataset_name, self.model_config)

        self.model_fn = model_fn
        self.server = FederatedServer(model_fn, val_dataset=test_dataset, device=device)

        self.clients: Dict[int, FederatedClient] = {}
        for cid in sorted(train_partitions.keys()):
            train_inds = train_partitions[cid]
            test_inds = test_partitions.get(cid, [])
            self.clients[cid] = FederatedClient(
                client_id=cid,
                dataset=train_dataset,
                train_indices=train_inds,
                test_indices=test_inds,
                model_fn=model_fn,
                device=device
            )

        self.num_clients = len(self.clients)
        logger.info(f"Initialized FL simulation with {self.num_clients} clients on device '{device}'.")

    def run(
        self,
        num_rounds: int = 10,
        local_epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01,
        optimizer_type: str = "SGD",
        client_fraction: float = 1.0,
        round_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None
    ) -> pd.DataFrame:
        """Execute federated learning simulation."""
        rng = np.random.RandomState(self.seed)
        round_logs: List[Dict[str, Any]] = []
        start_time = time.time()

        logger.info(f"Starting {num_rounds} communication rounds (C={client_fraction}, E={local_epochs}, B={batch_size}, lr={lr}).")

        for r in range(1, num_rounds + 1):
            if should_stop and should_stop():
                logger.warning(f"Training stopped prematurely at round {r} by user request.")
                break

            round_start = time.time()
            global_weights = self.server.get_global_weights()

            # Client selection
            if client_fraction >= 1.0:
                selected_cids = list(self.clients.keys())
            else:
                m = max(1, int(client_fraction * self.num_clients))
                selected_cids = rng.choice(list(self.clients.keys()), size=m, replace=False).tolist()

            client_updates = []
            train_losses = []
            train_accs = []
            total_train_samples = 0

            for cid in selected_cids:
                client = self.clients[cid]
                updated_weights, n_samples, loss, acc = client.train_one_round(
                    global_state_dict=global_weights,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    optimizer_type=optimizer_type
                )
                if n_samples > 0:
                    client_updates.append((updated_weights, n_samples))
                    train_losses.append(loss * n_samples)
                    train_accs.append(acc * n_samples)
                    total_train_samples += n_samples

            if client_updates:
                self.server.aggregate_updates(client_updates)

            eval_metrics = self.server.evaluate_global()
            round_elapsed = time.time() - round_start
            total_elapsed = time.time() - start_time

            avg_train_loss = float(sum(train_losses) / total_train_samples) if total_train_samples > 0 else 0.0
            avg_train_acc = float(sum(train_accs) / total_train_samples) if total_train_samples > 0 else 0.0

            log_entry = {
                "round": r,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "val_accuracy": float(eval_metrics.get("accuracy", 0.0)),
                "val_loss": float(eval_metrics.get("loss", 0.0)),
                "participating_clients": len(selected_cids),
                "round_time_sec": float(round_elapsed),
                "total_elapsed_sec": float(total_elapsed),
            }
            round_logs.append(log_entry)

            logger.info(
                f"Round {r:02d}/{num_rounds:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {eval_metrics.get('accuracy', 0.0):.4f} | "
                f"Time: {round_elapsed:.2f}s"
            )

            if round_callback:
                round_callback(log_entry)

        return pd.DataFrame(round_logs)