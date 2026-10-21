"""Page 4: Export Phase 1 Artifacts for Phase 2 Evaluation."""

from pathlib import Path
import streamlit as st
import pandas as pd

from ui.theme import apply_custom_theme
from ui.navigation import render_step_tracker
from ui.sidebar import render_sidebar
from ui.guards import render_guard_barrier
from ui.components import render_header, render_metric_card, render_download_button
from src.phase1_fl.export import export_phase1_run
from src.utils.artifact_manager import ArtifactManager

apply_custom_theme()
render_sidebar()
render_step_tracker(4)

if not render_guard_barrier(4, "Export Model"):
    st.stop()

render_header(
    title="Step 4: Export Phase 1 Output",
    subtitle="Export verified global model weights, per-client test partitions, and metadata to disk."
)

mgr: ArtifactManager = st.session_state.get("artifact_manager", ArtifactManager())

# Check if already exported or export now
if not st.session_state.get("phase1_exported", False):
    with st.spinner("Validating and writing Phase 1 artifacts to disk..."):
        dataset_name = st.session_state["dataset_name"]
        scenario = st.session_state["scenario"]
        exp_id = mgr.generate_experiment_id(prefix=f"{dataset_name.lower()}_{scenario.lower()}")

        export_res = export_phase1_run(
            dataset_name=dataset_name,
            scenario=scenario,
            experiment_id=exp_id,
            global_model=st.session_state["trained_global_model"],
            test_dataset=st.session_state["test_dataset"],
            test_partitions=st.session_state["test_partitions"],
            round_logs=st.session_state["round_logs"],
            partition_metadata=st.session_state["partition_metadata"],
            model_config=st.session_state["dataset_meta"],
            seed=st.session_state["seed"],
            evaluation_metric="accuracy",
            artifact_manager=mgr
        )

        st.session_state["phase1_export_result"] = export_res
        st.session_state["latest_experiment_id"] = exp_id
        st.session_state["phase1_exported"] = True

export_res = st.session_state["phase1_export_result"]
exp_dir = Path(export_res["directory"])
manifest = export_res["manifest"]
file_sizes = export_res["file_sizes"]

st.success(f"✅ Phase 1 experiment `{manifest['experiment_id']}` successfully exported and validated!")

# Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Export Status", "VALIDATED", "All checksums verified")
with col2:
    render_metric_card("Final Accuracy", f"{manifest['final_accuracy']:.4f}", f"Rounds: {manifest['num_rounds']}")
with col3:
    render_metric_card("Client Test Sets", str(manifest["client_test_sets_count"]), "Saved independently")
with col4:
    render_metric_card("Global Model", file_sizes["global_model.pt"], "PyTorch state_dict")

st.markdown("<br/>", unsafe_allow_html=True)

# Exported Files Table
st.markdown("### 📁 Exported Artifacts")
rows = [
    {"Artifact": "Global Model Weights", "Relative Path": "global_model.pt", "File Size": file_sizes["global_model.pt"]},
    {"Artifact": "Round Logs", "Relative Path": "round_logs.csv", "File Size": file_sizes["round_logs.csv"]},
    {"Artifact": "Model Architecture", "Relative Path": "model_config.json", "File Size": file_sizes["model_config.json"]},
    {"Artifact": "Partition Metadata", "Relative Path": "partition_metadata.json", "File Size": file_sizes["partition_metadata.json"]},
    {"Artifact": "Experiment Manifest", "Relative Path": "experiment_manifest.json", "File Size": file_sizes["experiment_manifest.json"]},
    {"Artifact": "Client Test Partitions", "Relative Path": "client_test_sets/*.pt", "File Size": f"{manifest['client_test_sets_count']} client files"},
]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown(f"**Absolute Directory:** `{exp_dir}`")

st.divider()

# Download artifacts
st.markdown("### ⬇️ Download Artifacts")
d_col1, d_col2, d_col3 = st.columns(3)
with d_col1:
    render_download_button(exp_dir / "round_logs.csv", label="Round Logs (CSV)", mime="text/csv")
with d_col2:
    render_download_button(exp_dir / "experiment_manifest.json", label="Manifest (JSON)", mime="application/json")
with d_col3:
    render_download_button(exp_dir / "global_model.pt", label="Model Weights (.pt)", mime="application/octet-stream")

st.divider()

# Transition to Phase 2
col_act1, col_act2 = st.columns([1, 2])
with col_act1:
    if st.button("🚀 Proceed to Phase 2: Statistical Evaluation", type="primary", use_container_width=True):
        st.session_state["phase2_loaded_exp_path"] = str(exp_dir)
        st.session_state["current_step"] = 5
        st.switch_page("pages/phase2_fed4fed/5_Load_Phase1_Output.py")
