"""End-to-end runner for Phase 2 FED4FED statistical evaluation pipeline."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import numpy as np
import pandas as pd
import torch

from src.models.registry import get_model
from src.phase2_fed4fed.bootstrap import run_clients_bootstrap
from src.phase2_fed4fed.fairness import evaluate_fairness
from src.reports.pdf_report import generate_pdf_report
from src.utils.artifact_manager import ArtifactManager
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("fed4fed.pipeline.phase2")


def run_phase2_pipeline(
    phase1_exp_dir: Path | str,
    num_bootstrap_resamples: int = 200,
    metric_name: str = "accuracy",
    significance_level: float = 0.05,
    confidence_level: float = 0.95,
    run_alg1: bool = True,
    run_alg2: bool = True,
    decision_policy: str = "conservative",
    seed: int = 42,
    device: str = "cpu",
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """Execute complete Phase 2 evaluation on an exported Phase 1 experiment."""
    set_seed(seed)
    mgr = ArtifactManager()
    exp_path = Path(phase1_exp_dir)

    # 1. Validate Phase 1 artifacts
    validation = mgr.validate_phase1_experiment(exp_path)
    if not validation["valid"]:
        raise ValueError(f"Cannot run Phase 2: {validation['error']}")

    manifest = validation["manifest"]
    dataset_name = manifest["dataset"]
    scenario = manifest["scenario"]
    exp_id = manifest["experiment_id"]
    model_cfg = validation["model_config"]

    logger.info(f"Loading Phase 1 artifacts for {exp_id} ({dataset_name}, {scenario})...")
    if progress_callback:
        progress_callback("Validating Phase 1 checkpoints and client test sets...", 0.1)

    # 2. Instantiate and load global model
    model = get_model(dataset_name, model_cfg).to(device)
    model_file = exp_path / "global_model.pt"
    state_dict = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load per-client private test sets and generate predictions
    test_dir = exp_path / "client_test_sets"
    test_files = sorted(test_dir.glob("client_*_test.pt"), key=lambda f: int(f.stem.split("_")[1]))
    
    if progress_callback:
        progress_callback("Evaluating global model across decentralized clients...", 0.3)

    client_test_pairs: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for tf in test_files:
            client_data = torch.load(tf, weights_only=False)
            cid = client_data["client_id"]
            c_x = client_data["data"]
            c_y = client_data["targets"]

            if len(c_y) == 0:
                continue

            c_x_dev = c_x.to(device)
            logits = model(c_x_dev)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = c_y.numpy()

            client_test_pairs[cid] = (targets, preds)

    # 4. Run Bootstrap Analysis
    if progress_callback:
        progress_callback(f"Computing {num_bootstrap_resamples} bootstrap resamples per client...", 0.5)

    boot_results = run_clients_bootstrap(
        client_test_data=client_test_pairs,
        num_resamples=num_bootstrap_resamples,
        metric_name=metric_name,
        seed=seed
    )

    # 5. Execute Fairness Testing and Confidence Intervals
    if progress_callback:
        progress_callback("Executing Algorithm 1, Algorithm 2, and Algorithm 3...", 0.7)

    fairness_results = evaluate_fairness(
        client_means=boot_results["bootstrap_means"],
        client_variances=boot_results["bootstrap_variances"],
        run_alg1=run_alg1,
        run_alg2=run_alg2,
        significance_level=significance_level,
        confidence_level=confidence_level,
        decision_policy=decision_policy
    )

    # 6. Save Phase 2 outputs
    out_dir = mgr.get_phase2_path(dataset_name, scenario, exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save bootstrap matrix
    np.save(out_dir / "bootstrap_data.npy", boot_results["bootstrap_data"])

    # Save client statistics CSV
    stats_rows = []
    for cid in sorted(client_test_pairs.keys()):
        c_boot = boot_results["clients"][cid]
        stats_rows.append({
            "client_id": cid,
            "sample_count": c_boot["sample_count"],
            "observed_metric": c_boot["observed"],
            "bootstrap_mean": c_boot["bootstrap_mean"],
            "bootstrap_variance": c_boot["bootstrap_variance"],
            "bootstrap_se": c_boot["bootstrap_se"],
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "client_statistics.csv", index=False)

    # Save pvalues and test results JSON
    pvals_json = {
        "experiment_id": exp_id,
        "dataset": dataset_name,
        "scenario": scenario,
        "timestamp": pd.Timestamp.now().isoformat(),
        "num_clients": len(client_test_pairs),
        "num_bootstrap_resamples": num_bootstrap_resamples,
        "significance_level": significance_level,
        "overall_verdict": fairness_results["overall_verdict"],
        "reject_h0": fairness_results["reject_h0"],
        "plain_english": fairness_results["plain_english"],
        "alg1_result": fairness_results["alg1_result"],
        "alg2_result": fairness_results["alg2_result"],
    }
    with open(out_dir / "pvalues.json", "w", encoding="utf-8") as f:
        json.dump(pvals_json, f, indent=2)

    # Save confidence intervals JSON
    with open(out_dir / "confidence_intervals.json", "w", encoding="utf-8") as f:
        json.dump(fairness_results["confidence_interval"], f, indent=2)

    # 7. Generate PDF Report
    if progress_callback:
        progress_callback("Compiling downloadable research PDF report...", 0.9)

    pdf_path = out_dir / "report.pdf"
    try:
        generate_pdf_report(
            output_pdf_path=pdf_path,
            manifest=manifest,
            partition_metadata=validation["partition_metadata"],
            round_logs=validation["logs_df"],
            bootstrap_summary=boot_results,
            fairness_results=fairness_results,
            client_statistics_df=stats_df
        )
    except Exception as e:
        logger.error(f"PDF report generation encountered an error: {e}")

    if progress_callback:
        progress_callback("Evaluation complete!", 1.0)

    logger.info(f"Phase 2 evaluation completed and saved to {out_dir}")
    return {
        "output_directory": str(out_dir),
        "bootstrap_results": boot_results,
        "fairness_results": fairness_results,
        "client_statistics_df": stats_df,
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
    }
