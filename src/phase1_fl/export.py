
"""Export engine for Phase 1 federated learning artifacts."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import TensorDatasetWrapper
from src.utils.artifact_manager import ArtifactManager
from src.utils.logger import get_logger

logger = get_logger("fed4fed.phase1.export")


def _json_default(obj):
    """Convert unsupported objects into JSON-compatible values."""

    # Handle vocabulary objects, including torchtext-style vocabularies.
    if hasattr(obj, "get_stoi") and callable(obj.get_stoi):
        return obj.get_stoi()

    # Handle objects with a dictionary conversion method.
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Handle NumPy arrays and scalar values.
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    # Handle sets, tuples, and paths.
    if isinstance(obj, (set, tuple)):
        return list(obj)

    if isinstance(obj, Path):
        return str(obj)

    # Handle PyTorch tensors.
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # Handle other objects exposing a state dictionary.
    if hasattr(obj, "__dict__"):
        return vars(obj)

    # Do not silently hide unsupported object types.
    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )


def export_phase1_run(
    dataset_name: str,
    scenario: str,
    experiment_id: str,
    global_model: torch.nn.Module,
    test_dataset: TensorDatasetWrapper,
    test_partitions: Dict[int, List[int]],
    round_logs: pd.DataFrame,
    partition_metadata: Dict[str, Any],
    model_config: Dict[str, Any],
    seed: int = 42,
    evaluation_metric: str = "accuracy",
    artifact_manager: ArtifactManager | None = None
) -> Dict[str, Any]:
    """Export all Phase 1 artifacts to disk for Phase 2 handoff."""

    mgr = artifact_manager or ArtifactManager()
    exp_dir = mgr.get_phase1_path(
        dataset_name, scenario, experiment_id
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    client_test_dir = exp_dir / "client_test_sets"
    client_test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save global model checkpoint.
    model_path = exp_dir / "global_model.pt"
    torch.save(global_model.state_dict(), model_path)

    # 2. Save per-client private test sets.
    client_test_summary = {}

    for cid, indices in test_partitions.items():
        if len(indices) > 0:
            client_data = test_dataset.data[indices]
            client_targets = test_dataset.targets[indices]
        else:
            client_data = torch.empty(0)
            client_targets = torch.empty(
                0, dtype=torch.long
            )

        c_file = client_test_dir / f"client_{cid}_test.pt"

        torch.save(
            {
                "client_id": cid,
                "data": client_data,
                "targets": client_targets,
                "sample_count": len(indices),
                "indices": indices,
            },
            c_file,
        )

        client_test_summary[str(cid)] = {
            "sample_count": len(indices),
            "file": c_file.name,
        }

    # 3. Save round logs.
    logs_path = exp_dir / "round_logs.csv"
    round_logs.to_csv(logs_path, index=False)

    # 4. Save model configuration.
    model_cfg_path = exp_dir / "model_config.json"

    with open(model_cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            model_config,
            f,
            indent=2,
            default=_json_default,
        )

    # 5. Save partition metadata.
    part_path = exp_dir / "partition_metadata.json"

    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(
            partition_metadata,
            f,
            indent=2,
            default=_json_default,
        )

    # 6. Save experiment manifest.
    final_val_acc = (
        float(round_logs["val_accuracy"].iloc[-1])
        if "val_accuracy" in round_logs.columns
        and len(round_logs) > 0
        else 0.0
    )

    final_train_loss = (
        float(round_logs["train_loss"].iloc[-1])
        if "train_loss" in round_logs.columns
        and len(round_logs) > 0
        else 0.0
    )

    manifest = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "scenario": scenario,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "evaluation_metric": evaluation_metric,
        "num_clients": len(test_partitions),
        "num_rounds": len(round_logs),
        "final_accuracy": final_val_acc,
        "final_loss": final_train_loss,
        "client_test_sets_count": len(client_test_summary),
        "files": {
            "model": str(model_path.relative_to(exp_dir)),
            "round_logs": str(logs_path.relative_to(exp_dir)),
            "model_config": str(
                model_cfg_path.relative_to(exp_dir)
            ),
            "partition_metadata": str(
                part_path.relative_to(exp_dir)
            ),
            "client_test_sets_dir": str(
                client_test_dir.relative_to(exp_dir)
            ),
        },
    }

    manifest_path = exp_dir / "experiment_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 7. Validate export completeness.
    validation = mgr.validate_phase1_experiment(exp_dir)

    if not validation["valid"]:
        raise RuntimeError(
            "Export validation failed: "
            f"{validation.get('error')}"
        )

    # 8. Compute artifact file sizes.
    file_sizes = {
        "global_model.pt": (
            f"{model_path.stat().st_size / (1024 * 1024):.2f} MB"
        ),
        "round_logs.csv": (
            f"{logs_path.stat().st_size / 1024:.2f} KB"
        ),
        "model_config.json": (
            f"{model_cfg_path.stat().st_size / 1024:.2f} KB"
        ),
        "partition_metadata.json": (
            f"{part_path.stat().st_size / 1024:.2f} KB"
        ),
        "experiment_manifest.json": (
            f"{manifest_path.stat().st_size / 1024:.2f} KB"
        ),
    }

    logger.info(
        f"Phase 1 experiment '{experiment_id}' "
        f"exported and verified at {exp_dir}"
    )

    return {
        "experiment_id": experiment_id,
        "directory": str(exp_dir),
        "manifest": manifest,
        "file_sizes": file_sizes,
        "validation": validation,
    }
