"""
=========================================================
DataPilot AI
Universal Machine Learning Engine
=========================================================

Supports:

• Classification
• Regression
• Cross Validation
• Model Comparison
• Feature Importance
• Model Persistence

Author: DataPilot AI
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

# Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

# Regression

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

# Metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.model_selection import (
    cross_val_score
)

# ======================================================
# MODEL REGISTRY
# ======================================================

CLASSIFICATION_MODELS = {

    "Logistic Regression":
        LogisticRegression(max_iter=500),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Extra Trees":
        ExtraTreesClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(),

    "Support Vector Machine":
        SVC(probability=True),

    "Gaussian Naive Bayes":
        GaussianNB()

}

REGRESSION_MODELS = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(random_state=42),

    "Random Forest":
        RandomForestRegressor(random_state=42),

    "Extra Trees":
        ExtraTreesRegressor(random_state=42),

    "Gradient Boosting":
        GradientBoostingRegressor(random_state=42),

    "KNN":
        KNeighborsRegressor(),

    "Support Vector Machine":
        SVR()

}

# ======================================================
# Train Single Model
# ======================================================

def train_model(

        model,

        X_train,

        y_train

):

    fitted = clone(model)

    fitted.fit(

        X_train,

        y_train

    )

    return fitted


# ======================================================
# Predict
# ======================================================

def predict(

        model,

        X_test

):

    return model.predict(X_test)

# ======================================================
# Classification Metrics
# ======================================================

def classification_metrics(

        y_true,

        y_pred

):

    return {

        "Accuracy":

            accuracy_score(

                y_true,

                y_pred

            ),

        "Precision":

            precision_score(

                y_true,

                y_pred,

                average="weighted",

                zero_division=0

            ),

        "Recall":

            recall_score(

                y_true,

                y_pred,

                average="weighted",

                zero_division=0

            ),

        "F1":

            f1_score(

                y_true,

                y_pred,

                average="weighted"

            )

    }


# ======================================================
# Regression Metrics
# ======================================================

def regression_metrics(

        y_true,

        y_pred

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

# ======================================================
# Cross Validation
# ======================================================

def cross_validate(

        model,

        X,

        y,

        scoring="accuracy",

        cv=5

):

    scores = cross_val_score(

        model,

        X,

        y,

        scoring=scoring,

        cv=cv

    )

    return {

        "Mean":

            scores.mean(),

        "Std":

            scores.std(),

        "Scores":

            scores

    }

# ======================================================
# Save Model
# ======================================================

def save_model(

        model,

        filename

):

    joblib.dump(

        model,

        filename

    )


# ======================================================
# Load Model
# ======================================================

def load_model(

        filename

):

    return joblib.load(

        filename

    )

