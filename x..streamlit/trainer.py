"""
=========================================================
DataPilot AI
Training Orchestrator
=========================================================

Automatically trains every supported model,
evaluates performance and produces a leaderboard.
"""

from __future__ import annotations

import pandas as pd

from core.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    train_model,
    predict,
    classification_metrics,
    regression_metrics,
    cross_validate,
)

def train_single_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type,
):
    """
    Train and evaluate a single model.
    """

    fitted = train_model(
        model,
        X_train,
        y_train
    )

    predictions = predict(
        fitted,
        X_test
    )

    if problem_type == "Classification":

        metrics = classification_metrics(
            y_test,
            predictions
        )

        cv = cross_validate(
            fitted,
            X_train,
            y_train,
            scoring="accuracy"
        )

    else:

        metrics = regression_metrics(
            y_test,
            predictions
        )

        cv = cross_validate(
            fitted,
            X_train,
            y_train,
            scoring="r2"
        )

    return {

        "name": name,

        "model": fitted,

        "metrics": metrics,

        "cv": cv,

        "predictions": predictions

    }

def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type="Classification",
):
    """
    Train every available model.
    """

    if problem_type == "Classification":

        registry = CLASSIFICATION_MODELS

        ranking_metric = "Accuracy"

    else:

        registry = REGRESSION_MODELS

        ranking_metric = "R²"

    trained = []

    leaderboard = []

    for name, model in registry.items():

        result = train_single_model(

            name,

            model,

            X_train,

            X_test,

            y_train,

            y_test,

            problem_type

        )

        trained.append(result)

        row = {

            "Model": name,

            **result["metrics"],

            "CV Mean":

                result["cv"]["Mean"],

            "CV Std":

                result["cv"]["Std"]

        }

        leaderboard.append(row)

    leaderboard = pd.DataFrame(
        leaderboard
    )

    leaderboard = leaderboard.sort_values(
        ranking_metric,
        ascending=False
    ).reset_index(drop=True)

    best_name = leaderboard.iloc[0]["Model"]

    best_model = next(

        model["model"]

        for model in trained

        if model["name"] == best_name

    )

    return {

        "leaderboard": leaderboard,

        "best_model": best_model,

        "results": trained

    }

def feature_importance(
    model,
    feature_names,
):
    """
    Return feature importance if supported.
    """

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance":

                model.feature_importances_

        })

        return importance.sort_values(

            "Importance",

            ascending=False

        )

    if hasattr(model, "coef_"):

        coef = model.coef_

        if coef.ndim > 1:

            coef = coef.mean(axis=0)

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": abs(coef)

        })

        return importance.sort_values(

            "Importance",

            ascending=False

        )

    return None

def summarize_training(results):
    """
    Create a concise training summary.
    """

    leaderboard = results["leaderboard"]

    best = leaderboard.iloc[0]

    return {

        "Best Model": best["Model"],

        "Cross Validation":

            round(

                best["CV Mean"],

                4

            ),

        "Leaderboard":

            leaderboard

    }

