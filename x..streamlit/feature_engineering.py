"""
=========================================================
DataPilot AI
Feature Engineering Engine
=========================================================

Automatic feature engineering and feature selection
for universal tabular datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures


# ==========================================================
# Low Variance Features
# ==========================================================

def low_variance_columns(
        df: pd.DataFrame,
        threshold: float = 0.01
):

    numeric = df.select_dtypes(include=np.number)

    selector = VarianceThreshold(
        threshold=threshold
    )

    selector.fit(numeric)

    keep = numeric.columns[
        selector.get_support()
    ]

    remove = [
        col
        for col in numeric.columns
        if col not in keep
    ]

    return remove


# ==========================================================
# Highly Correlated Features
# ==========================================================

def highly_correlated_columns(
        df,
        threshold=0.95
):

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr().abs()

    upper = corr.where(

        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)

    )

    remove = [

        column

        for column in upper.columns

        if any(
            upper[column] > threshold
        )

    ]

    return remove


# ==========================================================
# Remove Features
# ==========================================================

def remove_columns(
        df,
        columns
):

    columns = [

        c for c in columns

        if c in df.columns

    ]

    return df.drop(
        columns=columns
    )

# ==========================================================
# Polynomial Features
# ==========================================================

def polynomial_features(
        df,
        degree=2
):

    numeric = df.select_dtypes(
        include=np.number
    )

    transformer = PolynomialFeatures(

        degree=degree,

        include_bias=False

    )

    data = transformer.fit_transform(
        numeric
    )

    names = transformer.get_feature_names_out(
        numeric.columns
    )

    return (

        pd.DataFrame(
            data,
            columns=names,
            index=df.index
        ),

        transformer

    )


# ==========================================================
# Log Transform
# ==========================================================

def log_transform(

        df,

        columns

):

    df = df.copy()

    for column in columns:

        if column in df.columns:

            values = df[column]

            if (values > 0).all():

                df[column] = np.log1p(values)

    return df


# ==========================================================
# Datetime Expansion
# ==========================================================

def expand_datetime_features(df):

    df = df.copy()

    datetime_columns = df.select_dtypes(

        include=["datetime64"]

    ).columns

    for column in datetime_columns:

        df[f"{column}_year"] = df[column].dt.year

        df[f"{column}_month"] = df[column].dt.month

        df[f"{column}_day"] = df[column].dt.day

        df[f"{column}_weekday"] = df[column].dt.weekday

    return df

# ==========================================================
# ANOVA Feature Selection
# ==========================================================

def anova_selection(

        X,

        y,

        k=10

):

    selector = SelectKBest(

        score_func=f_classif,

        k=min(k, X.shape[1])

    )

    selector.fit(X, y)

    features = X.columns[

        selector.get_support()

    ]

    scores = selector.scores_

    return pd.DataFrame({

        "Feature": X.columns,

        "Score": scores

    }).sort_values(

        "Score",

        ascending=False

    )


# ==========================================================
# Mutual Information
# ==========================================================

def mutual_information_selection(

        X,

        y

):

    scores = mutual_info_classif(

        X,

        y

    )

    return pd.DataFrame({

        "Feature": X.columns,

        "Score": scores

    }).sort_values(

        "Score",

        ascending=False

    )

# ==========================================================
# Recursive Feature Elimination
# ==========================================================

def rfe_selection(

        X,

        y,

        n_features=10

):

    estimator = RandomForestClassifier(

        random_state=42

    )

    selector = RFE(

        estimator,

        n_features_to_select=min(

            n_features,

            X.shape[1]

        )

    )

    selector.fit(

        X,

        y

    )

    selected = X.columns[

        selector.support_

    ]

    return list(selected)


# ==========================================================
# Intelligent Recommendations
# ==========================================================

def recommend_feature_engineering(df):

    recommendations = []

    low_variance = low_variance_columns(df)

    if low_variance:

        recommendations.append(

            f"Remove {len(low_variance)} low variance features."

        )

    correlated = highly_correlated_columns(df)

    if correlated:

        recommendations.append(

            f"Remove {len(correlated)} highly correlated features."

        )

    datetime_columns = df.select_dtypes(

        include=["datetime64"]

    ).columns

    if len(datetime_columns):

        recommendations.append(

            "Expand datetime features."

        )

    numeric = df.select_dtypes(

        include=np.number

    )

    if len(numeric.columns):

        recommendations.append(

            "Consider polynomial features."

        )

    return recommendations

