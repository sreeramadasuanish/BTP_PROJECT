"""Streamlit session state management for FED4FED platform."""

from typing import Any, Dict, Optional
import streamlit as st

from src.utils.artifact_manager import ArtifactManager


def init_session_state() -> None:
    """Initialize all FED4FED session state variables with clean defaults."""
    defaults = {
        # Navigation / Step Tracking
        "current_step": 0,

        # Phase 1: Selection & Partitioning
        "dataset_name": "MNIST",
        "scenario": "Fair",
        "num_clients": 100,
        "seed": 42,
        "device": "cpu",
        "train_dataset": None,
        "test_dataset": None,
        "dataset_meta": None,
        "dataset_loaded": False,

        # Partitions
        "train_partitions": None,
        "test_partitions": None,
        "partition_metadata": None,
        "partitions_created": False,

        # Phase 1: Training
        "num_rounds": 5,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "optimizer_type": "SGD",
        "client_fraction": 1.0,
        "training_completed": False,
        "round_logs": None,
        "trained_global_model": None,

        # Phase 1: Export
        "phase1_exported": False,
        "phase1_export_result": None,
        "latest_experiment_id": None,

        # Phase 2: Loading
        "phase2_loaded_exp_path": None,
        "phase2_loaded_manifest": None,
        "phase2_loaded_validation": None,
        "phase2_is_loaded": False,

        # Phase 2: Bootstrap
        "bootstrap_resamples": 200,
        "bootstrap_metric": "accuracy",
        "bootstrap_results": None,
        "bootstrap_completed": False,

        # Phase 2: Fairness & Confidence Interval
        "significance_level": 0.05,
        "confidence_level": 0.95,
        "run_alg1": True,
        "run_alg2": True,
        "decision_policy": "conservative",
        "fairness_results": None,
        "fairness_completed": False,

        # Artifact Manager
        "artifact_manager": ArtifactManager(),
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_state(key: str, default: Any = None) -> Any:
    """Safely get session state value."""
    return st.session_state.get(key, default)


def set_state(key: str, val: Any) -> None:
    """Safely set session state value."""
    st.session_state[key] = val
