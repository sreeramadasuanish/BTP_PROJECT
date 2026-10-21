"""End-to-end runner for Phase 1 Federated Learning pipeline."""

from typing import Any, Callable, Dict, Optional
import pandas as pd

from src.data.registry import load_dataset
from src.data.partition import create_partitions
from src.phase1_fl.simulation import FLSimulation
from src.phase1_fl.export import export_phase1_run
from src.utils.artifact_manager import ArtifactManager
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("fed4fed.pipeline.phase1")


def run_phase1_pipeline(
    dataset_name: str = "MNIST",
    scenario: str = "Fair",
    num_clients: int = 100,
    num_rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    optimizer_type: str = "SGD",
    client_fraction: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    partition_kwargs: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    round_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 1 pipeline: Load -> Partition -> Train -> Export."""
    set_seed(seed)
    mgr = ArtifactManager()
    exp_id = experiment_id or mgr.generate_experiment_id(prefix=f"{dataset_name.lower()}_{scenario.lower()}")

    logger.info(f"Starting Phase 1 pipeline for {dataset_name} ({scenario} scenario, {num_clients} clients)...")

    # 1. Load dataset
    train_ds, test_ds, meta = load_dataset(dataset_name)

    # 2. Create partitions
    p_kwargs = partition_kwargs or {}
    train_parts, test_parts, part_meta = create_partitions(
        dataset_name=dataset_name,
        scenario=scenario,
        train_targets=train_ds.targets.numpy(),
        test_targets=test_ds.targets.numpy(),
        num_clients=num_clients,
        seed=seed,
        **p_kwargs
    )

    # 3. Initialize simulation
    sim = FLSimulation(
        dataset_name=dataset_name,
        train_dataset=train_ds,
        test_dataset=test_ds,
        train_partitions=train_parts,
        test_partitions=test_parts,
        model_config=meta,
        device=device,
        seed=seed
    )

    # 4. Train federated model
    round_logs = sim.run(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_type=optimizer_type,
        client_fraction=client_fraction,
        round_callback=round_callback,
        should_stop=should_stop
    )

    # 5. Export Phase 1 artifacts
    export_result = export_phase1_run(
        dataset_name=dataset_name,
        scenario=scenario,
        experiment_id=exp_id,
        global_model=sim.server.global_model,
        test_dataset=test_ds,
        test_partitions=test_parts,
        round_logs=round_logs,
        partition_metadata=part_meta,
        model_config=meta,
        seed=seed,
        evaluation_metric="accuracy",
        artifact_manager=mgr
    )

    logger.info(f"Phase 1 pipeline completed successfully for {exp_id}.")
    return {
        "experiment_id": exp_id,
        "round_logs": round_logs,
        "export_result": export_result,
        "partition_metadata": part_meta,
        "model_config": meta,
        "simulation": sim,
    }
