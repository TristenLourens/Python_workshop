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

core/loader.py

uploaded = files.upload()

filename = list(uploaded.keys())[0]

df = pd.read_csv(filename, sep=";")

  """
=========================================================
DataPilot AI
Universal Dataset Loader
=========================================================
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st


SUPPORTED_EXTENSIONS = (
    ".csv",
    ".xlsx",
    ".xls",
)

MAX_FILE_SIZE_MB = 20


# -------------------------------------------------------
# File Validation
# -------------------------------------------------------

def validate_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded file.
    """

    if uploaded_file is None:
        return False, "No file selected."

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:

        return (
            False,
            f"Unsupported file type: {suffix}"
        )

    size_mb = uploaded_file.size / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:

        return (
            False,
            f"File exceeds {MAX_FILE_SIZE_MB} MB."
        )

    return True, ""


# -------------------------------------------------------
# Detect CSV Delimiter
# -------------------------------------------------------

def detect_delimiter(uploaded_file) -> str:

    uploaded_file.seek(0)

    sample = uploaded_file.read(4096).decode(
        "utf-8",
        errors="ignore"
    )

    uploaded_file.seek(0)

    try:

        dialect = csv.Sniffer().sniff(sample)

        return dialect.delimiter

    except Exception:

        return ","


# -------------------------------------------------------
# Read Dataset
# -------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file):

    suffix = Path(uploaded_file.name).suffix.lower()

    try:

        if suffix == ".csv":

            delimiter = detect_delimiter(uploaded_file)

            df = pd.read_csv(
                uploaded_file,
                sep=delimiter,
                engine="python"
            )

        elif suffix in (".xlsx", ".xls"):

            df = pd.read_excel(uploaded_file)

        else:

            raise ValueError("Unsupported format")

        return df

    except UnicodeDecodeError:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            encoding="latin1"
        )

        return df

    except Exception as e:

        st.error(e)

        return None


# -------------------------------------------------------
# Dataset Summary
# -------------------------------------------------------

def dataset_summary(df):

    summary = {

        "Rows": df.shape[0],

        "Columns": df.shape[1],

        "Missing Values":
            int(df.isna().sum().sum()),

        "Duplicate Rows":
            int(df.duplicated().sum()),

        "Memory (MB)":
            round(
                df.memory_usage(
                    deep=True
                ).sum() / 1024**2,
                2
            )

    }

    return summary

core/inference.py

numeric_features = [...]

categorical_features = [...]

core/inference.py

"""
=========================================================
DataPilot AI
Automatic Dataset Inference Engine
=========================================================
"""

from __future__ import annotations

import pandas as pd


# -------------------------------------------------------
# Infer Column Types
# -------------------------------------------------------

def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Automatically infer the type of every column.
    """

    numeric = []
    categorical = []
    datetime_cols = []
    boolean = []
    text = []
    identifier = []

    total_rows = len(df)

    for column in df.columns:

        series = df[column]

        # ---------- Boolean ----------

        if series.dtype == "bool":

            boolean.append(column)

            continue

        unique_values = series.nunique(dropna=True)

        # ---------- Identifier ----------

        if unique_values == total_rows:

            if "id" in column.lower():

                identifier.append(column)

                continue

        # ---------- Numeric ----------

        if pd.api.types.is_numeric_dtype(series):

            numeric.append(column)

            continue

        # ---------- Datetime ----------

        try:

            converted = pd.to_datetime(series)

            if converted.notna().sum() > total_rows * 0.8:

                datetime_cols.append(column)

                continue

        except Exception:

            pass

        # ---------- Text ----------

        if pd.api.types.is_string_dtype(series):

            avg_length = series.astype(str).str.len().mean()

            if avg_length > 40:

                text.append(column)

            else:

                categorical.append(column)

        else:

            categorical.append(column)

    return {

        "numeric": numeric,

        "categorical": categorical,

        "datetime": datetime_cols,

        "boolean": boolean,

        "text": text,

        "identifier": identifier

    }


# -------------------------------------------------------
# Candidate Target Columns
# -------------------------------------------------------

def suggest_target_columns(df: pd.DataFrame):

    """
    Return likely target variables.
    """

    candidates = []

    for column in df.columns:

        unique = df[column].nunique()

        if 2 <= unique <= 20:

            candidates.append(column)

    return candidates


# -------------------------------------------------------
# Infer ML Problem Type
# -------------------------------------------------------

def infer_problem_type(df: pd.DataFrame,
                       target_column: str):

    """
    Determine whether this is a
    classification or regression task.
    """

    y = df[target_column]

    if pd.api.types.is_numeric_dtype(y):

        if y.nunique() <= 20:

            return "Classification"

        return "Regression"

    return "Classification"


# -------------------------------------------------------
# Dataset Health Score
# -------------------------------------------------------

def calculate_health_score(df: pd.DataFrame):

    """
    Returns a simple quality score (0-100)
    """

    score = 100

    # Missing values

    missing_ratio = df.isna().sum().sum() / df.size

    score -= missing_ratio * 40

    # Duplicate rows

    duplicate_ratio = df.duplicated().mean()

    score -= duplicate_ratio * 30

    # Constant columns

    constant_columns = 0

    for col in df.columns:

        if df[col].nunique(dropna=False) <= 1:

            constant_columns += 1

    if len(df.columns) > 0:

        score -= (constant_columns / len(df.columns)) * 30

    score = max(0, min(100, round(score)))

    return score


# -------------------------------------------------------
# Dataset Summary
# -------------------------------------------------------

def dataset_overview(df: pd.DataFrame):

    info = {}

    info["Rows"] = df.shape[0]

    info["Columns"] = df.shape[1]

    info["Missing Values"] = int(
        df.isna().sum().sum()
    )

    info["Duplicate Rows"] = int(
        df.duplicated().sum()
    )

    info["Memory (MB)"] = round(
        df.memory_usage(
            deep=True
        ).sum() / (1024 ** 2),
        2
    )

    info["Health Score"] = calculate_health_score(df)

    return info

    core/session.py

    """
=========================================================
DataPilot AI
Session State Manager
=========================================================
"""

from __future__ import annotations

import streamlit as st


# -------------------------------------------------------
# Default Session State
# -------------------------------------------------------

DEFAULT_STATE = {

    # Dataset
    "dataset": None,
    "filename": None,

    # Dataset Information
    "summary": None,
    "health_score": None,

    # Column Types
    "numeric_columns": [],
    "categorical_columns": [],
    "datetime_columns": [],
    "boolean_columns": [],
    "text_columns": [],
    "identifier_columns": [],

    # Target
    "target": None,
    "problem_type": None,

    # Processed Data
    "cleaned_dataset": None,
    "X": None,
    "y": None,

    # Preprocessing
    "preprocessor": None,
    "encoded_features": None,

    # Feature Engineering
    "selected_features": None

    ,
    # Machine Learning
    "trained_models": {},
    "best_model": None,
    "leaderboard": None,

    # Evaluation
    "predictions": None,
    "metrics": None,

    # Explainability
    "shap_values": None,

    # Reports
    "report": None,

    # Flags
    "dataset_loaded": False,
    "preprocessing_complete": False,
    "training_complete": False
}


# -------------------------------------------------------
# Initialise Session
# -------------------------------------------------------

def initialise_session():

    """
    Initialise Streamlit session state.
    """

    for key, value in DEFAULT_STATE.items():

        if key not in st.session_state:

            st.session_state[key] = value


# -------------------------------------------------------
# Reset Dataset
# -------------------------------------------------------

def clear_dataset():

    """
    Remove all dataset-related objects.
    """

    dataset_keys = [

        "dataset",
        "filename",
        "summary",
        "health_score",
        "numeric_columns",
        "categorical_columns",
        "datetime_columns",
        "boolean_columns",
        "text_columns",
        "identifier_columns",
        "target",
        "problem_type",
        "cleaned_dataset",
        "X",
        "y",
        "preprocessor",
        "encoded_features",
        "selected_features",
        "trained_models",
        "best_model",
        "leaderboard",
        "predictions",
        "metrics",
        "shap_values",
        "report",
        "dataset_loaded",
        "preprocessing_complete",
        "training_complete"

    ]

    for key in dataset_keys:

        st.session_state[key] = DEFAULT_STATE[key]


# -------------------------------------------------------
# Dataset Exists
# -------------------------------------------------------

def has_dataset():

    return st.session_state.dataset is not None


# -------------------------------------------------------
# Training Exists
# -------------------------------------------------------

def has_model():

    return st.session_state.best_model is not None


# -------------------------------------------------------
# Update Dataset
# -------------------------------------------------------

def update_dataset(

        df,

        filename,

        summary,

        inferred_types,

        health_score

):

    st.session_state.dataset = df

    st.session_state.filename = filename

    st.session_state.summary = summary

    st.session_state.health_score = health_score

    st.session_state.numeric_columns = inferred_types["numeric"]

    st.session_state.categorical_columns = inferred_types["categorical"]

    st.session_state.datetime_columns = inferred_types["datetime"]

    st.session_state.boolean_columns = inferred_types["boolean"]

    st.session_state.text_columns = inferred_types["text"]

    st.session_state.identifier_columns = inferred_types["identifier"]

    st.session_state.dataset_loaded = True


# -------------------------------------------------------
# Update Target
# -------------------------------------------------------

def update_target(

        target,

        problem_type

):

    st.session_state.target = target

    st.session_state.problem_type = problem_type


# -------------------------------------------------------
# Save Trained Model
# -------------------------------------------------------

def save_model(

        name,

        model

):

    st.session_state.trained_models[name] = model

    st.session_state.best_model = model

    st.session_state.training_complete = True


# -------------------------------------------------------
# Save Metrics
# -------------------------------------------------------

def save_metrics(metrics):

    st.session_state.metrics = metrics


# -------------------------------------------------------
# Save Leaderboard
# -------------------------------------------------------

def save_leaderboard(df):

    st.session_state.leaderboard = df

    pages/
    01_Home.py

    """
=========================================================
DataPilot AI
Home Dashboard
=========================================================
"""

from pathlib import Path

import streamlit as st

from core.session import initialise_session, has_dataset

# --------------------------------------------------------
# Initialise Session
# --------------------------------------------------------

initialise_session()

# --------------------------------------------------------
# Page Config
# --------------------------------------------------------

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------------
# CSS
# --------------------------------------------------------

st.markdown(
    """
<style>

.main-title{
    font-size:52px;
    font-weight:800;
}

.subtitle{
    color:gray;
    font-size:18px;
}

.metric{
    background:#1a1d24;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# Header
# --------------------------------------------------------

st.markdown(
    """
<div class="main-title">

🚀 DataPilot AI

</div>

<div class="subtitle">

An Intelligent Machine Learning Workbench

</div>

""",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------
# Dataset Status
# --------------------------------------------------------

st.subheader("Dataset Status")

if has_dataset():

    df = st.session_state.dataset

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with c2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:

        st.metric(
            "Health Score",
            f"{st.session_state.health_score}%"
        )

    with c4:

        st.metric(
            "Problem Type",
            st.session_state.problem_type
            if st.session_state.problem_type
            else "Not Selected"
        )

else:

    st.info(
        "No dataset loaded."
    )

# --------------------------------------------------------
# Workflow
# --------------------------------------------------------

st.divider()

st.subheader("Workflow")

workflow = [

    "📂 Upload",

    "📊 Explore",

    "🧹 Clean",

    "⚙ Feature Engineering",

    "🤖 Train",

    "🚀 AutoML",

    "📈 Evaluate",

    "🧠 Explain",

    "📄 Report"

]

cols = st.columns(len(workflow))

for col, step in zip(cols, workflow):

    col.markdown(
        f"<center>{step}</center>",
        unsafe_allow_html=True
    )

# --------------------------------------------------------
# About
# --------------------------------------------------------

st.divider()

st.subheader("About")

st.write("""

DataPilot AI is designed to analyse any structured dataset.

Features include

- Automatic dataset profiling

- Exploratory data analysis

- Data cleaning

- Feature engineering

- Machine learning

- AutoML

- Explainable AI

- Report generation

Upload a CSV or Excel dataset to begin.

""")

# --------------------------------------------------------
# Footer
# --------------------------------------------------------

st.divider()

st.caption("DataPilot AI • Version 1.0")

pages/02_Upload.py

"""
=========================================================
DataPilot AI
Upload Dataset
=========================================================
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from core.loader import (
    validate_file,
    load_dataset,
    dataset_summary
)

from core.inference import (
    infer_column_types,
    calculate_health_score,
    suggest_target_columns,
)

from core.session import (
    initialise_session,
    update_dataset,
)

# -------------------------------------------------------
# Initialise
# -------------------------------------------------------

initialise_session()

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📂",
    layout="wide"
)

# -------------------------------------------------------
# CSS
# -------------------------------------------------------

st.markdown("""
<style>

.upload-box{
    border:2px dashed #4F8BF9;
    border-radius:15px;
    padding:30px;
    text-align:center;
    background:#161b22;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("📂 Upload Dataset")

st.caption(
    "Upload a CSV or Excel dataset (maximum 20 MB)"
)

st.divider()

# -------------------------------------------------------
# Upload
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Drag and drop your dataset",
    type=["csv", "xlsx", "xls"]
)

# -------------------------------------------------------
# Process Upload
# -------------------------------------------------------

if uploaded_file:

    valid, message = validate_file(uploaded_file)

    if not valid:

        st.error(message)

        st.stop()

    with st.spinner("Loading dataset..."):

        df = load_dataset(uploaded_file)

    if df is None:

        st.error("Dataset could not be loaded.")

        st.stop()

    # ----------------------------------------------
    # Infer Dataset
    # ----------------------------------------------

    inferred = infer_column_types(df)

    summary = dataset_summary(df)

    health = calculate_health_score(df)

    update_dataset(
        df=df,
        filename=uploaded_file.name,
        summary=summary,
        inferred_types=inferred,
        health_score=health
    )

    st.success("Dataset loaded successfully!")

    st.divider()

    # ----------------------------------------------
    # Summary Cards
    # ----------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", summary["Rows"])
    c2.metric("Columns", summary["Columns"])
    c3.metric("Missing Values", summary["Missing Values"])
    c4.metric("Health Score", f"{health}%")

    st.divider()

    # ----------------------------------------------
    # Column Types
    # ----------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Detected Features")

        st.write(
            f"**Numeric:** {len(inferred['numeric'])}"
        )

        st.write(
            f"**Categorical:** {len(inferred['categorical'])}"
        )

        st.write(
            f"**Datetime:** {len(inferred['datetime'])}"
        )

        st.write(
            f"**Boolean:** {len(inferred['boolean'])}"
        )

        st.write(
            f"**Text:** {len(inferred['text'])}"
        )

        st.write(
            f"**Identifiers:** {len(inferred['identifier'])}"
        )

    with right:

        st.subheader("Suggested Target Columns")

        targets = suggest_target_columns(df)

        if len(targets):

            target = st.selectbox(
                "Choose Target Variable",
                targets
            )

            st.session_state.target = target

            st.success(
                f"Target selected: {target}"
            )

        else:

            st.warning(
                "No suitable target columns detected."
            )

    st.divider()

    # ----------------------------------------------
    # Preview
    # ----------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        height=450
    )

    # ----------------------------------------------
    # Data Types
    # ----------------------------------------------

    with st.expander("Column Information"):

        info = pd.DataFrame({

            "Column": df.columns,

            "Data Type": df.dtypes.astype(str),

            "Missing":

                df.isna().sum(),

            "Unique":

                df.nunique()

        })

        st.dataframe(
            info,
            use_container_width=True
        )

else:

    st.info(
        "Please upload a CSV or Excel file to begin."
    )

    core/
    visualizations.py

    """
=========================================================
DataPilot AI
Visualization Library
=========================================================
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# -------------------------------------------------------
# Missing Values
# -------------------------------------------------------

def missing_values_chart(df: pd.DataFrame):

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        return None

    fig = px.bar(
        missing.sort_values(ascending=False),
        title="Missing Values by Column",
        labels={
            "value": "Missing Values",
            "index": "Column"
        }
    )

    fig.update_layout(height=500)

    return fig


# -------------------------------------------------------
# Correlation Matrix
# -------------------------------------------------------

def correlation_heatmap(df: pd.DataFrame):

    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix"
    )

    fig.update_layout(height=700)

    return fig


# -------------------------------------------------------
# Histogram
# -------------------------------------------------------

def histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=40,
        title=f"Distribution of {column}"
    )

    fig.update_layout(height=500)

    return fig


# -------------------------------------------------------
# Boxplot
# -------------------------------------------------------

def boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        points="outliers",
        title=f"Boxplot of {column}"
    )

    fig.update_layout(height=500)

    return fig


# -------------------------------------------------------
# Scatter Plot
# -------------------------------------------------------

def scatter(df, x, y, colour=None):

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=colour
    )

    fig.update_layout(height=550)

    return fig


# -------------------------------------------------------
# Bar Chart
# -------------------------------------------------------

def categorical_counts(df, column):

    counts = df[column].value_counts()

    fig = px.bar(
        counts,
        title=f"{column} Distribution"
    )

    fig.update_layout(height=500)

    return fig


# -------------------------------------------------------
# Correlation Table
# -------------------------------------------------------

def strongest_correlations(df):

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr().abs()

    pairs = (
        corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1
            ).astype(bool)
        )
        .stack()
        .reset_index()
    )

    pairs.columns = [

        "Feature A",

        "Feature B",

        "Correlation"

    ]

    pairs = pairs.sort_values(
        "Correlation",
        ascending=False
    )

    return pairs

    pages/
    03_Explore.py

    """
=========================================================
DataPilot AI
Explore Dataset
=========================================================
"""

from __future__ import annotations

import streamlit as st

from core.session import initialise_session
from core.visualizations import *

initialise_session()

st.set_page_config(
    page_title="Explore Dataset",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")

if st.session_state.dataset is None:

    st.warning(
        "Please upload a dataset first."
    )

    st.stop()

df = st.session_state.dataset

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------

overview, distributions, relationships, quality = st.tabs(

    [

        "Overview",

        "Distributions",

        "Relationships",

        "Data Quality"

    ]

)

# ====================================================
# OVERVIEW
# ====================================================

with overview:

    st.subheader("Dataset Overview")

    st.dataframe(df.head())

    st.write(df.describe(include="all"))

# ====================================================
# DISTRIBUTIONS
# ====================================================

with distributions:

    numeric = st.session_state.numeric_columns

    categorical = st.session_state.categorical_columns

    if numeric:

        feature = st.selectbox(

            "Numeric Variable",

            numeric

        )

        st.plotly_chart(

            histogram(df, feature),

            use_container_width=True

        )

        st.plotly_chart(

            boxplot(df, feature),

            use_container_width=True

        )

    if categorical:

        category = st.selectbox(

            "Categorical Variable",

            categorical

        )

        st.plotly_chart(

            categorical_counts(df, category),

            use_container_width=True

        )

# ====================================================
# RELATIONSHIPS
# ====================================================

with relationships:

    st.plotly_chart(

        correlation_heatmap(df),

        use_container_width=True

    )

    numeric = st.session_state.numeric_columns

    if len(numeric) >= 2:

        x = st.selectbox(

            "X Axis",

            numeric,

            key="scatter_x"

        )

        y = st.selectbox(

            "Y Axis",

            numeric,

            index=1,

            key="scatter_y"

        )

        st.plotly_chart(

            scatter(df, x, y),

            use_container_width=True

        )

# ====================================================
# DATA QUALITY
# ====================================================

with quality:

    fig = missing_values_chart(df)

    if fig:

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    else:

        st.success(
            "No missing values detected."
        )

    st.subheader("Strongest Correlations")

    st.dataframe(

        strongest_correlations(df).head(25),

        use_container_width=True

    )

    pages/04_Clean.py

    """
=========================================================
DataPilot AI
Data Cleaning & Preprocessing
=========================================================
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core.session import initialise_session

initialise_session()

st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Cleaning")

if st.session_state.dataset is None:

    st.warning("Please upload a dataset first.")

    st.stop()

df = st.session_state.dataset.copy()

# ======================================================
# Missing Values
# ======================================================

st.header("Missing Values")

missing = df.isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    st.success("No missing values detected.")

else:

    st.dataframe(
        missing.to_frame("Missing Values"),
        use_container_width=True
    )

    option = st.selectbox(

        "Missing Value Strategy",

        [

            "Do Nothing",

            "Drop Rows",

            "Mean",

            "Median",

            "Most Frequent"

        ]

    )

    if st.button("Apply Missing Value Strategy"):

        if option == "Drop Rows":

            df = df.dropna()

        elif option == "Mean":

            numeric = df.select_dtypes(include="number").columns

            imputer = SimpleImputer(strategy="mean")

            df[numeric] = imputer.fit_transform(df[numeric])

        elif option == "Median":

            numeric = df.select_dtypes(include="number").columns

            imputer = SimpleImputer(strategy="median")

            df[numeric] = imputer.fit_transform(df[numeric])

        elif option == "Most Frequent":

            imputer = SimpleImputer(strategy="most_frequent")

            df[:] = imputer.fit_transform(df)

        st.success("Missing values handled.")

# ======================================================
# Duplicate Rows
# ======================================================

st.divider()

st.header("Duplicate Rows")

duplicates = df.duplicated().sum()

st.metric("Duplicate Rows", duplicates)

if duplicates > 0:

    if st.button("Remove Duplicates"):

        df = df.drop_duplicates()

        st.success("Duplicates removed.")

# ======================================================
# Encoding
# ======================================================

st.divider()

st.header("Categorical Encoding")

categorical = df.select_dtypes(include="object").columns

if len(categorical):

    selected = st.multiselect(

        "Columns",

        categorical,

        default=list(categorical)

    )

    if st.button("Encode Selected Columns"):

        for col in selected:

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(

                df[col].astype(str)

            )

        st.success("Encoding complete.")

else:

    st.info("No categorical columns detected.")

# ======================================================
# Scaling
# ======================================================

st.divider()

st.header("Scaling")

numeric = df.select_dtypes(include="number").columns

scale = st.checkbox("Scale numeric features")

if scale:

    scaler = StandardScaler()

    df[numeric] = scaler.fit_transform(df[numeric])

    st.success("Scaling complete.")

# ======================================================
# Save
# ======================================================

st.divider()

if st.button("Save Cleaned Dataset"):

    st.session_state.cleaned_dataset = df

    st.session_state.dataset = df

    st.session_state.preprocessing_complete = True

    st.success("Dataset updated successfully.")

# ======================================================
# Preview
# ======================================================

st.divider()

st.subheader("Preview")

st.dataframe(

    df.head(),

    use_container_width=True
)

st.write(f"Shape: {df.shape}")

pages/
    04_Clean.py          # UI only
    05_Feature_Engineering.py
    06_Train.py

core/
    cleaner.py           # All preprocessing functions
    feature_engineering.py
    models.py

    
