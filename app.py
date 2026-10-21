"""Main Streamlit application entrypoint for FED4FED research dashboard."""

import streamlit as st
from ui.state import init_session_state
from ui.theme import apply_custom_theme

# Page configuration
st.set_page_config(
    page_title="FED4FED | Federated Learning & Fairness Suite",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and styling
init_session_state()
apply_custom_theme()

# Define explicit navigation pages
pages = {
    "Overview": [
        st.Page("pages/0_Home_Overview.py", title="Home Overview", icon="🏠", default=True),
    ],
    "Phase 1: Federated Learning": [
        st.Page("pages/phase1_fl/1_Select_Dataset.py", title="1. Select Dataset", icon="📊"),
        st.Page("pages/phase1_fl/2_Partition_Clients.py", title="2. Partition Clients", icon="👥"),
        st.Page("pages/phase1_fl/3_FL_Training.py", title="3. FL Training", icon="⚙️"),
        st.Page("pages/phase1_fl/4_Export_Model.py", title="4. Export Model", icon="💾"),
    ],
    "Phase 2: FED4FED Statistical Evaluation": [
        st.Page("pages/phase2_fed4fed/5_Load_Phase1_Output.py", title="5. Load Phase 1 Output", icon="📂"),
        st.Page("pages/phase2_fed4fed/6_Bootstrap.py", title="6. Bootstrap Analysis", icon="🎲"),
        st.Page("pages/phase2_fed4fed/7_Fairness_Test.py", title="7. Fairness Test", icon="⚖️"),
        st.Page("pages/phase2_fed4fed/8_Confidence_Interval.py", title="8. Confidence Interval", icon="📏"),
        st.Page("pages/phase2_fed4fed/9_Final_Report.py", title="9. Final Report", icon="📑"),
    ]
}

pg = st.navigation(pages)
pg.run()
