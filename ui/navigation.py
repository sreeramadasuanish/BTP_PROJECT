"""Workflow step tracker and navigation components."""

from typing import List, Tuple
import streamlit as st

STEPS: List[Tuple[int, str, str]] = [
    (0, "0. Home", "Home Overview"),
    (1, "1. Dataset", "Phase 1: Select Dataset"),
    (2, "2. Partition", "Phase 1: Partition Clients"),
    (3, "3. Train FL", "Phase 1: FL Training"),
    (4, "4. Export", "Phase 1: Export Model"),
    (5, "5. Load FL", "Phase 2: Load Phase 1"),
    (6, "6. Bootstrap", "Phase 2: Bootstrap"),
    (7, "7. Fairness", "Phase 2: Fairness Test"),
    (8, "8. Confidence", "Phase 2: Confidence Interval"),
    (9, "9. Report", "Phase 2: Final Report"),
]


def render_step_tracker(current_step: int) -> None:
    """Render interactive visual stepper across Phase 1 and Phase 2."""
    cols = st.columns(len(STEPS))
    for idx, (step_num, label, desc) in enumerate(STEPS):
        with cols[idx]:
            is_active = (step_num == current_step)
            # Determine if step is completed
            is_completed = False
            if step_num == 1 and st.session_state.get("dataset_loaded"):
                is_completed = True
            elif step_num == 2 and st.session_state.get("partitions_created"):
                is_completed = True
            elif step_num == 3 and st.session_state.get("training_completed"):
                is_completed = True
            elif step_num == 4 and st.session_state.get("phase1_exported"):
                is_completed = True
            elif step_num == 5 and st.session_state.get("phase2_is_loaded"):
                is_completed = True
            elif step_num == 6 and st.session_state.get("bootstrap_completed"):
                is_completed = True
            elif step_num == 7 and st.session_state.get("fairness_completed"):
                is_completed = True
            elif step_num == 8 and st.session_state.get("fairness_completed"):
                is_completed = True

            css_class = "step-item"
            if is_active:
                css_class += " active"
            elif is_completed:
                css_class += " completed"

            badge = "●" if is_active else ("✓" if is_completed else "○")
            st.markdown(
                f"<div class='{css_class}' title='{desc}'>{badge} {label}</div>",
                unsafe_allow_html=True
            )
