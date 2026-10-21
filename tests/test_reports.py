"""Unit tests for research PDF report and summary table generation."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.reports.pdf_report import generate_pdf_report
from src.reports.result_tables import build_fairness_summary_table, build_client_summary_table


def test_pdf_report_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_file = Path(tmp_dir) / "test_report.pdf"

        manifest = {
            "experiment_id": "test_exp_999",
            "dataset": "MNIST",
            "scenario": "Fair",
            "num_clients": 5,
            "num_rounds": 3,
            "final_accuracy": 0.88,
            "seed": 42
        }
        part_meta = {"method": "uniform"}
        round_logs = pd.DataFrame([{"round": 1, "train_loss": 0.5, "val_accuracy": 0.85}])
        boot_summary = {"num_resamples": 100}

        fairness_results = {
            "overall_verdict": "FAIR (Performance Homogeneous)",
            "reject_h0": False,
            "plain_english": "Model performance is statistically homogeneous across clients.",
            "alg1_result": {
                "test_statistic": 1.234,
                "degrees_of_freedom": 4,
                "p_value": 0.873,
                "significance_level": 0.05,
                "reject_h0": False
            },
            "confidence_interval": {
                "is_permitted": True,
                "confidence_level": 0.95,
                "pooled_estimate": 0.88,
                "pooled_ci_lower": 0.85,
                "pooled_ci_upper": 0.91,
                "pooled_se": 0.015,
                "effective_df": 15.0
            }
        }

        client_df = pd.DataFrame([
            {"client_id": i, "sample_count": 50, "observed_metric": 0.88, "bootstrap_mean": 0.88, "bootstrap_variance": 0.001, "bootstrap_se": 0.031}
            for i in range(5)
        ])

        generate_pdf_report(
            output_pdf_path=pdf_file,
            manifest=manifest,
            partition_metadata=part_meta,
            round_logs=round_logs,
            bootstrap_summary=boot_summary,
            fairness_results=fairness_results,
            client_statistics_df=client_df
        )

        assert pdf_file.exists()
        assert pdf_file.stat().st_size > 1000, "PDF file is too small or empty"
