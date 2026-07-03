"""
=========================================================
DataPilot AI
Data Cleaning Engine
=========================================================

This module contains reusable preprocessing functions
that can be used throughout the application.

Author: DataPilot AI
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder


# =========================================================
# Missing Values
# =========================================================

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe describing missing values.
    """

    summary = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isna().sum(),

        "Missing (%)":
            (df.isna().mean()*100).round(2)

    })

    summary = summary[
        summary["Missing Values"] > 0
    ]

    return summary.sort_values(
        "Missing (%)",
        ascending=False
    )


# =========================================================
# Drop Missing Rows
# =========================================================

def drop_missing_rows(df):

    return df.dropna().copy()


# =========================================================
# Drop Missing Columns
# =========================================================

def drop_missing_columns(
        df,
        threshold=50
):

    """
    Drops columns exceeding the threshold.

    threshold = percentage
    """

    percent = df.isna().mean()*100

    cols = percent[
        percent > threshold
    ].index

    return df.drop(columns=cols)


# =========================================================
# Imputation
# =========================================================

def impute_numeric(
        df,
        strategy="mean"
):

    df = df.copy()

    numeric = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric):

        imputer = SimpleImputer(
            strategy=strategy
        )

        df[numeric] = imputer.fit_transform(
            df[numeric]
        )

    return df


def impute_categorical(df):

    df = df.copy()

    categorical = df.select_dtypes(
        exclude=np.number
    ).columns

    if len(categorical):

        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        df[categorical] = imputer.fit_transform(
            df[categorical]
        )

    return df


# =========================================================
# Duplicate Rows
# =========================================================

def duplicate_summary(df):

    return int(df.duplicated().sum())


def remove_duplicates(df):

    return df.drop_duplicates().copy()


# =========================================================
# Constant Columns
# =========================================================

def constant_columns(df):

    cols = []

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:

            cols.append(column)

    return cols


def remove_constant_columns(df):

    cols = constant_columns(df)

    return df.drop(columns=cols)


# =========================================================
# Label Encoding
# =========================================================

def label_encode(df):

    df = df.copy()

    encoders = {}

    categorical = df.select_dtypes(
        include=["object","category"]
    ).columns

    for column in categorical:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

        encoders[column] = encoder

    return df, encoders


# =========================================================
# Scaling
# =========================================================

def scale_data(
        df,
        method="standard"
):

    df = df.copy()

    numeric = df.select_dtypes(
        include=np.number
    ).columns

    if method == "standard":

        scaler = StandardScaler()

    elif method == "minmax":

        scaler = MinMaxScaler()

    elif method == "robust":

        scaler = RobustScaler()

    else:

        raise ValueError(
            "Unknown scaling method."
        )

    df[numeric] = scaler.fit_transform(
        df[numeric]
    )

    return df, scaler


# =========================================================
# Full Cleaning Pipeline
# =========================================================

def clean_dataset(
        df,
        numeric_strategy="mean",
        remove_dupes=True,
        remove_constants=True,
        encode=True,
        scaling=None
):
    """
    Complete preprocessing pipeline.
    """

    df = df.copy()

    # Missing Values

    df = impute_numeric(
        df,
        strategy=numeric_strategy
    )

    df = impute_categorical(df)

    # Duplicates

    if remove_dupes:

        df = remove_duplicates(df)

    # Constant Features

    if remove_constants:

        df = remove_constant_columns(df)

    # Encoding

    encoders = {}

    if encode:

        df, encoders = label_encode(df)

    # Scaling

    scaler = None

    if scaling is not None:

        df, scaler = scale_data(
            df,
            method=scaling
        )

    return {

        "data": df,

        "encoders": encoders,

        "scaler": scaler

    }

result = clean_dataset(
    df,
    numeric_strategy="median",
    remove_dupes=True,
    encode=True,
    scaling="standard"
)

clean_df = result["data"]

