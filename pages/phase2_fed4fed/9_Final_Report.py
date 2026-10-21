"""Page 9: Final Research Report Generation and Artifact Download."""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_verdict_banner, render_download_button
from src.reports.pdf_report import generate_pdf_report
from src.reports.result_tables import build_fairness_summary_table
from src.utils.artifact_manager import ArtifactManager

apply_custom_theme()
render_sidebar()
render_step_tracker(9)

if not render_guard_barrier(9, "Final Report"):
    st.stop()

render_header(
    title="Step 9: Final Research Report",
    subtitle="Download comprehensive PDF research report, statistical summaries, CSV tables, and JSON metrics."
)

mgr: ArtifactManager = st.session_state.get("artifact_manager", ArtifactManager())
validation = st.session_state["phase2_loaded_validation"]
manifest = validation["manifest"]
dataset_name = manifest["dataset"]
scenario = manifest["scenario"]
exp_id = manifest["experiment_id"]
part_meta = validation["partition_metadata"]
round_logs = validation["logs_df"]

boot_res = st.session_state["bootstrap_results"]
fairness_res = st.session_state["fairness_results"]

# Output directory for Phase 2
out_dir = mgr.get_phase2_path(dataset_name, scenario, exp_id)
out_dir.mkdir(parents=True, exist_ok=True)

# Build client statistics DataFrame
stats_rows = []
for cid in range(boot_res["num_clients"]):
    c_boot = boot_res["clients"][cid]
    stats_rows.append({
        "client_id": cid,
        "sample_count": c_boot["sample_count"],
        "observed_metric": c_boot["observed"],
        "bootstrap_mean": c_boot["bootstrap_mean"],
        "bootstrap_variance": c_boot["bootstrap_variance"],
        "bootstrap_se": c_boot["bootstrap_se"],
    })
client_stats_df = pd.DataFrame(stats_rows)

# Generate PDF report
pdf_path = out_dir / "report.pdf"
try:
    generate_pdf_report(
        output_pdf_path=pdf_path,
        manifest=manifest,
        partition_metadata=part_meta,
        round_logs=round_logs,
        bootstrap_summary=boot_res,
        fairness_results=fairness_res,
        client_statistics_df=client_stats_df
    )
except Exception as e:
    st.error(f"Error generating PDF report: {e}")

# Save CSV and JSON
stats_csv_path = out_dir / "client_statistics.csv"
client_stats_df.to_csv(stats_csv_path, index=False)

pvals_json_path = out_dir / "pvalues.json"
pvals_data = {
    "experiment_id": exp_id,
    "dataset": dataset_name,
    "scenario": scenario,
    "overall_verdict": fairness_res["overall_verdict"],
    "reject_h0": fairness_res["reject_h0"],
    "plain_english": fairness_res["plain_english"],
    "alg1_result": fairness_res.get("alg1_result"),
    "alg2_result": fairness_res.get("alg2_result"),
}
with open(pvals_json_path, "w", encoding="utf-8") as f:
    json.dump(pvals_data, f, indent=2)

# Executive Verdict
render_verdict_banner(
    verdict=fairness_res["overall_verdict"],
    text=fairness_res["plain_english"],
    is_unfair=fairness_res["reject_h0"]
)

st.markdown("<br/>", unsafe_allow_html=True)

# Download Controls Grid
st.markdown("### 📥 Download Research Artifacts")
d_col1, d_col2, d_col3 = st.columns(3)

with d_col1:
    if pdf_path.exists():
        render_download_button(pdf_path, label="Final Report (PDF)", mime="application/pdf")
with d_col2:
    if stats_csv_path.exists():
        render_download_button(stats_csv_path, label="Client Stats (CSV)", mime="text/csv")
with d_col3:
    if pvals_json_path.exists():
        render_download_button(pvals_json_path, label="Statistical Results (JSON)", mime="application/json")

st.divider()

# Summary Tables
st.markdown("### 📑 Statistical Summary Tables")
st.dataframe(build_fairness_summary_table(fairness_res), use_container_width=True)

st.markdown("#### Decentralized Client Performance Table (Preview)")
st.dataframe(client_stats_df.head(15), use_container_width=True)

st.divider()

# Start over or return to home
col_end1, col_end2 = st.columns([1, 2])
with col_end1:
    if st.button("🏠 Return to Home Dashboard", type="primary", use_container_width=True):
        st.session_state["current_step"] = 0
        st.switch_page("pages/0_Home_Overview.py")
