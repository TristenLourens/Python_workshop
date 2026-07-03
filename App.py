from pathlib import Path
import os

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

ROOT = Path(__file__).parent

FOLDERS = {
    "assets": ROOT / "assets",
    "config": ROOT / "config",
    "core": ROOT / "core",
    "pages": ROOT / "pages",
    "models": ROOT / "models",
    "reports": ROOT / "reports",
    "data": ROOT / "data",
    "streamlit": ROOT / ".streamlit",
}

# Create folders if they don't exist
for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

# Optional shortcuts
ASSETS = FOLDERS["assets"]
CONFIG = FOLDERS["config"]
CORE = FOLDERS["core"]
PAGES = FOLDERS["pages"]
MODELS = FOLDERS["models"]
REPORTS = FOLDERS["reports"]
DATA = FOLDERS["data"]
STREAMLIT = FOLDERS["streamlit"]

=========================================================
DataPilot AI
An Intelligent Machine Learning Workbench
Version: 1.0
=========================================================
"""

from pathlib import Path

import streamlit as st
import pandas as pd

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

ROOT = Path(__file__).parent

ASSETS = ROOT / "assets"

DATA = ROOT / "data"

MODELS = ROOT / "models"

REPORTS = ROOT / "reports"

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------

st.markdown(
    """
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.main-title{
    font-size:52px;
    font-weight:800;
    margin-bottom:0;
}

.sub-title{
    color:#AAAAAA;
    font-size:18px;
    margin-top:-5px;
}

.metric-card{
    background:#1b1f2a;
    padding:20px;
    border-radius:18px;
    box-shadow:0 0 15px rgba(0,0,0,.25);
}

.section-card{
    background:#171a23;
    padding:25px;
    border-radius:20px;
}

.small-text{
    color:#9e9e9e;
    font-size:14px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:13px;
    padding-top:30px;
}

hr{
    margin-top:30px;
    margin-bottom:30px;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Session State
# -------------------------------------------------------

DEFAULT_STATE = {

    "dataset": None,

    "filename": None,

    "problem_type": None,

    "target": None,

    "numeric_columns": [],

    "categorical_columns": [],

    "datetime_columns": [],

    "boolean_columns": [],

    "text_columns": [],

    "id_columns": [],

    "health_score": None

}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.markdown(
    """
<div class="main-title">
🚀 DataPilot AI
</div>

<div class="sub-title">
Universal Machine Learning Studio
</div>
""",
    unsafe_allow_html=True
)

st.write("")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.image(
        "https://raw.githubusercontent.com/streamlit/brand/master/logos/mark/streamlit-mark-color.png",
        width=90,
    )

    st.title("Navigation")

    st.success("DataPilot AI v1.0")

    st.divider()

    st.page_link(
        "pages/01_Home.py",
        label="🏠 Home"
    )

    st.page_link(
        "pages/02_Upload.py",
        label="📂 Upload Dataset"
    )

    st.page_link(
        "pages/03_Explore.py",
        label="📊 Explore"
    )

    st.page_link(
        "pages/04_Clean.py",
        label="🧹 Clean"
    )

    st.page_link(
        "pages/05_Feature_Engineering.py",
        label="⚙ Feature Engineering"
    )

    st.page_link(
        "pages/06_Train.py",
        label="🤖 Train Models"
    )

    st.page_link(
        "pages/07_AutoML.py",
        label="🚀 AutoML"
    )

    st.page_link(
        "pages/08_Evaluate.py",
        label="📈 Evaluation"
    )

    st.page_link(
        "pages/09_Explain.py",
        label="🧠 Explainability"
    )

    st.page_link(
        "pages/10_Report.py",
        label="📄 Reports"
    )

    st.divider()

    if st.session_state.filename is not None:

        st.success("Dataset Loaded")

        st.caption(st.session_state.filename)

    else:

        st.warning("No Dataset Loaded")

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def metric_card(title, value):

    st.markdown(
        f"""
<div class="metric-card">
<h4>{title}</h4>
<h2>{value}</h2>
</div>
""",
        unsafe_allow_html=True
    )
# =====================================================
# MAIN DASHBOARD
# =====================================================

st.divider()

# -----------------------------------------------------
# Workflow Progress
# -----------------------------------------------------

workflow_steps = [
    "📂 Upload",
    "📊 Explore",
    "🧹 Clean",
    "⚙ Engineer",
    "🤖 Train",
    "📈 Evaluate",
    "🧠 Explain",
    "📄 Report"
]

st.subheader("Workflow")

progress = 0

if st.session_state.dataset is not None:
    progress = 12

st.progress(progress)

cols = st.columns(len(workflow_steps))

for col, step in zip(cols, workflow_steps):
    col.markdown(
        f"<center>{step}</center>",
        unsafe_allow_html=True
    )

st.write("")

# =====================================================
# DATASET STATUS
# =====================================================

st.subheader("Current Dataset")

if st.session_state.dataset is None:

    st.info(
        """
No dataset has been uploaded yet.

Click **Upload Dataset** in the sidebar to begin.
"""
    )

else:

    df = st.session_state.dataset

    rows = df.shape[0]
    cols = df.shape[1]

    numeric = len(st.session_state.numeric_columns)
    categorical = len(st.session_state.categorical_columns)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Rows",
            f"{rows:,}"
        )

    with col2:
        metric_card(
            "Columns",
            cols
        )

    with col3:
        metric_card(
            "Numeric",
            numeric
        )

    with col4:
        metric_card(
            "Categorical",
            categorical
        )

    st.write("")

    st.subheader("Preview")

    st.dataframe(
        df.head(),
        use_container_width=True,
        height=250
    )

# =====================================================
# QUICK ACTIONS
# =====================================================

st.divider()

st.subheader("Quick Actions")

c1, c2, c3, c4 = st.columns(4)

with c1:

    if st.button(
        "📂 Upload Dataset",
        use_container_width=True
    ):
        st.switch_page("pages/02_Upload.py")

with c2:

    if st.button(
        "📊 Explore Data",
        use_container_width=True,
        disabled=st.session_state.dataset is None
    ):
        st.switch_page("pages/03_Explore.py")

with c3:

    if st.button(
        "🤖 Train Model",
        use_container_width=True,
        disabled=st.session_state.dataset is None
    ):
        st.switch_page("pages/06_Train.py")

with c4:

    if st.button(
        "🚀 AutoML",
        use_container_width=True,
        disabled=st.session_state.dataset is None
    ):
        st.switch_page("pages/07_AutoML.py")

# =====================================================
# ABOUT
# =====================================================

st.divider()

left, right = st.columns([2,1])

with left:

    st.markdown("## Welcome to DataPilot AI")

    st.write(
        """
DataPilot AI is an intelligent machine learning workbench
designed to analyse **any tabular dataset**.

The platform automatically performs:

- Dataset profiling
- Exploratory Data Analysis
- Data cleaning
- Feature engineering
- Machine Learning
- AutoML
- Explainable AI
- Report generation

Simply upload your dataset to begin.
"""
    )

with right:

    st.markdown("### Features")

    st.success("✔ Universal CSV Support")

    st.success("✔ Excel Support")

    st.success("✔ Automatic Type Detection")

    st.success("✔ Interactive Visualizations")

    st.success("✔ AutoML")

    st.success("✔ Explainable AI")

    st.success("✔ Download Reports")

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
<div class="footer">

Made with ❤️ using Streamlit

</div>
""",
    unsafe_allow_html=True
)
