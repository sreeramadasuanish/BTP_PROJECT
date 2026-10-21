"""Unit tests for Phase 1 artifact exporting, loading, and validation."""

import tempfile
from pathlib import Path
import pytest
import torch
import pandas as pd

from src.models.mlp_mnist import MLPMnist
from src.data.preprocessing import TensorDatasetWrapper
from src.phase1_fl.export import export_phase1_run
from src.utils.artifact_manager import ArtifactManager


def test_export_and_validate_phase1_artifacts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = ArtifactManager(base_dir=Path(tmp_dir))

        model = MLPMnist()
        test_data = torch.randn(20, 1, 28, 28)
        test_targets = torch.randint(0, 10, (20,))
        test_ds = TensorDatasetWrapper(test_data, test_targets)
        test_parts = {0: list(range(10)), 1: list(range(10, 20))}
        logs_df = pd.DataFrame([{"round": 1, "train_loss": 0.5, "val_accuracy": 0.8}])

        export_res = export_phase1_run(
            dataset_name="MNIST",
            scenario="Fair",
            experiment_id="test_exp_001",
            global_model=model,
            test_dataset=test_ds,
            test_partitions=test_parts,
            round_logs=logs_df,
            partition_metadata={"method": "uniform"},
            model_config={"input_dim": 784},
            artifact_manager=mgr
        )

        assert export_res["validation"]["valid"] is True

        exp_dir = Path(export_res["directory"])
        assert (exp_dir / "global_model.pt").exists()
        assert (exp_dir / "round_logs.csv").exists()
        assert (exp_dir / "client_test_sets" / "client_0_test.pt").exists()
        assert (exp_dir / "client_test_sets" / "client_1_test.pt").exists()

        # Test validation fails if model file is removed
        (exp_dir / "global_model.pt").unlink()
        val_failed = mgr.validate_phase1_experiment(exp_dir)
        assert val_failed["valid"] is False
