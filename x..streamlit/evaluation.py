"""
=========================================================
DataPilot AI
Model Evaluation Engine
=========================================================

Reusable evaluation functions for both
classification and regression models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ==========================================================
# Confusion Matrix
# ==========================================================

def confusion_matrix_plot(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix"
    )

    fig.update_layout(
        height=600
    )

    return fig

# ==========================================================
# ROC Curve
# ==========================================================

def roc_curve_plot(
    model,
    X_test,
    y_test,
):

    if not hasattr(model, "predict_proba"):

        return None

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=fpr,

            y=tpr,

            mode="lines",

            name=f"AUC = {roc_auc:.3f}"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=[0, 1],

            y=[0, 1],

            mode="lines",

            line=dict(dash="dash"),

            showlegend=False

        )

    )

    fig.update_layout(

        title="ROC Curve",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate"

    )

    return fig

# ==========================================================
# Precision Recall Curve
# ==========================================================

def precision_recall_plot(
    model,
    X_test,
    y_test,
):

    if not hasattr(model, "predict_proba"):

        return None

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    precision, recall, _ = precision_recall_curve(

        y_test,

        probabilities

    )

    ap = average_precision_score(

        y_test,

        probabilities

    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=recall,

            y=precision,

            mode="lines",

            name=f"AP = {ap:.3f}"

        )

    )

    fig.update_layout(

        title="Precision Recall Curve",

        xaxis_title="Recall",

        yaxis_title="Precision"

    )

    return fig

# ==========================================================
# Actual vs Predicted
# ==========================================================

def actual_vs_predicted(
    y_true,
    y_pred,
):

    fig = px.scatter(

        x=y_true,

        y=y_pred,

        labels={

            "x": "Actual",

            "y": "Predicted"

        },

        title="Actual vs Predicted"

    )

    fig.add_shape(

        type="line",

        x0=min(y_true),

        x1=max(y_true),

        y0=min(y_true),

        y1=max(y_true)

    )

    return fig


# ==========================================================
# Residual Plot
# ==========================================================

def residual_plot(
    y_true,
    y_pred,
):

    residuals = y_true - y_pred

    fig = px.scatter(

        x=y_pred,

        y=residuals,

        labels={

            "x": "Predicted",

            "y": "Residual"

        },

        title="Residual Plot"

    )

    fig.add_hline(y=0)

    return fig
  # ==========================================================
# Classification Report
# ==========================================================

def classification_summary(
    y_true,
    y_pred,
):

    report = classification_report(

        y_true,

        y_pred,

        output_dict=True

    )

    return pd.DataFrame(report).transpose()


# ==========================================================
# Regression Summary
# ==========================================================

def regression_summary(
    y_true,
    y_pred,
):

    rmse = np.sqrt(

        mean_squared_error(

            y_true,

            y_pred

        )

    )

    return {

        "R²":

            r2_score(

                y_true,

                y_pred

            ),

        "RMSE":

            rmse,

        "MAE":

            mean_absolute_error(

                y_true,

                y_pred

            )

    }
