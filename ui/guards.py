"""Workflow prerequisite guards and validation barriers."""

from typing import Tuple
import streamlit as st


def check_guard(step_number: int) -> Tuple[bool, str]:
    """Check whether prerequisites for a given step are fulfilled.
    
    Returns:
        (is_allowed, error_message_if_not)
    """
    # Step 0: Always allowed
    if step_number == 0:
        return True, ""

    # Step 1: Select Dataset (Always accessible)
    if step_number == 1:
        return True, ""

    # Step 2: Partition Clients -> Requires Dataset loaded in Step 1
    if step_number == 2:
        if not st.session_state.get("dataset_loaded", False):
            return False, "Please select and load a dataset in Step 1 first."
        return True, ""

    # Step 3: FL Training -> Requires Partitions generated in Step 2
    if step_number == 3:
        if not st.session_state.get("partitions_created", False):
            return False, "Please generate client partitions in Step 2 before starting federated training."
        return True, ""

    # Step 4: Export Model -> Requires Training finished in Step 3
    if step_number == 4:
        if not st.session_state.get("training_completed", False):
            return False, "Please complete federated model training in Step 3 before exporting artifacts."
        return True, ""

    # Step 5: Load Phase 1 Output -> Always accessible to select existing or new run
    if step_number == 5:
        return True, ""

    # Step 6: Bootstrap -> Requires valid Phase 1 run loaded in Step 5
    if step_number == 6:
        if not st.session_state.get("phase2_is_loaded", False):
            return False, "Please select and load a completed Phase 1 experiment in Step 5 before running bootstrap."
        return True, ""

    # Step 7: Fairness Test -> Requires Bootstrap completed in Step 6
    if step_number == 7:
        if not st.session_state.get("bootstrap_completed", False):
            return False, "Please run the bootstrap analysis in Step 6 before conducting fairness hypothesis tests."
        return True, ""

    # Step 8: Confidence Interval -> Requires Fairness tests evaluated in Step 7
    if step_number == 8:
        if not st.session_state.get("fairness_completed", False):
            return False, "Please execute the fairness test in Step 7 before evaluating confidence intervals."
        return True, ""

    # Step 9: Final Report -> Requires evaluation evaluated in Step 7
    if step_number == 9:
        if not st.session_state.get("fairness_completed", False):
            return False, "Please complete the statistical fairness evaluation before generating the final report."
        return True, ""

    return True, ""


def render_guard_barrier(step_number: int, step_name: str) -> bool:
    """Render warning panel if guard condition fails, halting page rendering."""
    allowed, message = check_guard(step_number)
    if not allowed:
        st.warning(f"⚠️ **Prerequisite Missing for {step_name}**")
        st.info(message)
        st.markdown(
            "Use the sidebar navigation or step tracker above to visit the prerequisite step."
        )
        return False
    return True
