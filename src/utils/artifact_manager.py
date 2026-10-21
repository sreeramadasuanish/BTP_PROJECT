"""Experiment and artifact manager for Phase 1 and Phase 2 outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch

from src.utils.config import get_base_dir


class ArtifactManager:
    """Manages saving, loading, listing, and verifying FED4FED experiment artifacts."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or get_base_dir()
        self.phase1_dir = self.base_dir / "outputs" / "phase1"
        self.phase2_dir = self.base_dir / "outputs" / "phase2"
        self.phase1_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_experiment_id(prefix: str = "exp") -> str:
        """Generate unique timestamp-based experiment id."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def get_phase1_path(self, dataset: str, scenario: str, experiment_id: str) -> Path:
        """Return directory path for Phase 1 experiment."""
        return self.phase1_dir / dataset.lower().replace("-", "_") / scenario.lower() / experiment_id

    def get_phase2_path(self, dataset: str, scenario: str, experiment_id: str) -> Path:
        """Return directory path for Phase 2 experiment."""
        return self.phase2_dir / dataset.lower().replace("-", "_") / scenario.lower() / experiment_id

    def list_phase1_experiments(self) -> List[Dict[str, Any]]:
        """List all completed Phase 1 experiments with metadata."""
        experiments = []
        if not self.phase1_dir.exists():
            return experiments

        for manifest_path in self.phase1_dir.glob("*/*/*/experiment_manifest.json"):
            exp_dir = manifest_path.parent
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                # Check required files
                req_files = [
                    "global_model.pt",
                    "round_logs.csv",
                    "partition_metadata.json",
                    "model_config.json"
                ]
                is_valid = all((exp_dir / rf).exists() for rf in req_files) and (exp_dir / "client_test_sets").exists()
                
                manifest["is_valid"] = is_valid
                manifest["path"] = str(exp_dir)
                experiments.append(manifest)
            except Exception:
                continue

        # Sort by timestamp descending
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return experiments

    def validate_phase1_experiment(self, exp_dir: Path | str) -> Dict[str, Any]:
        """Validate Phase 1 artifacts and return summary or raise exception."""
        p = Path(exp_dir)
        if not p.exists():
            return {"valid": False, "error": f"Experiment directory does not exist: {p}"}

        manifest_file = p / "experiment_manifest.json"
        model_file = p / "global_model.pt"
        logs_file = p / "round_logs.csv"
        part_file = p / "partition_metadata.json"
        model_cfg_file = p / "model_config.json"
        test_sets_dir = p / "client_test_sets"

        for file_path, name in [
            (manifest_file, "Manifest"),
            (model_file, "Global Model Checkpoint"),
            (logs_file, "Round Logs"),
            (part_file, "Partition Metadata"),
            (model_cfg_file, "Model Config"),
        ]:
            if not file_path.exists():
                return {"valid": False, "error": f"Missing required file: {name} ({file_path.name})"}

        if not test_sets_dir.exists() or not any(test_sets_dir.glob("*.pt")):
            return {"valid": False, "error": f"Missing or empty client test sets directory: {test_sets_dir}"}

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            with open(part_file, "r", encoding="utf-8") as f:
                partition_meta = json.load(f)
            with open(model_cfg_file, "r", encoding="utf-8") as f:
                model_cfg = json.load(f)
            logs_df = pd.read_csv(logs_file)

            test_files = list(test_sets_dir.glob("client_*_test.pt"))
            num_clients = len(test_files)

            return {
                "valid": True,
                "manifest": manifest,
                "partition_metadata": partition_meta,
                "model_config": model_cfg,
                "num_clients": num_clients,
                "num_rounds": len(logs_df),
                "final_accuracy": logs_df["val_accuracy"].iloc[-1] if "val_accuracy" in logs_df.columns else None,
                "final_loss": logs_df["train_loss"].iloc[-1] if "train_loss" in logs_df.columns else None,
                "logs_df": logs_df,
                "path": str(p),
            }
        except Exception as e:
            return {"valid": False, "error": f"Failed to parse experiment artifacts: {e}"}

    def list_phase2_evaluations(self) -> List[Dict[str, Any]]:
        """List all completed Phase 2 statistical evaluations."""
        evals = []
        if not self.phase2_dir.exists():
            return evals

        for pvals_path in self.phase2_dir.glob("*/*/*/pvalues.json"):
            exp_dir = pvals_path.parent
            try:
                with open(pvals_path, "r", encoding="utf-8") as f:
                    pvals = json.load(f)
                ci_path = exp_dir / "confidence_intervals.json"
                cis = {}
                if ci_path.exists():
                    with open(ci_path, "r", encoding="utf-8") as f:
                        cis = json.load(f)

                evals.append({
                    "path": str(exp_dir),
                    "experiment_id": exp_dir.name,
                    "scenario": exp_dir.parent.name,
                    "dataset": exp_dir.parent.parent.name,
                    "pvalues": pvals,
                    "confidence_intervals": cis,
                    "has_report": (exp_dir / "report.pdf").exists()
                })
            except Exception:
                continue

        evals.sort(key=lambda x: x.get("pvalues", {}).get("timestamp", ""), reverse=True)
        return evals
