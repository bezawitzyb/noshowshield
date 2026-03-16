"""
NoShowShield — Sklearn preprocessing pipeline.

Responsibilities:
    - Build a ColumnTransformer that scales numericals and encodes categoricals
    - Consistent transformation between training and testing data
    - Prevent data leakage: fit only on training data, transform on both

Usage:
    from noshowshield.eda_package.preprocessor import build_preprocessor

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer

def group_countries(
    data:pd.DataFrame,
    limit:int
    )-> pd.DataFrame:

    country_counts = data['country'].value_counts()
    countries_included = country_counts[country_counts >= limit].index

    data['country_group'] = data['country'].apply(
        lambda x: x if x in countries_included else 'Other'
    )
    return data

def engineer_numerical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from the raw hotel booking dataset.

    This function:
    - removes the target if it is present
    - removes known leakage columns
    - converts ID-like columns into binary indicators
    - creates additional behavioral features

    Parameters
    ----------
    X : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features added.
    """
    X = X.copy()

    # Drop target if it is accidentally included
    if "is_canceled" in X.columns:
        X = X.drop(columns="is_canceled")

    # Drop leakage columns: these contain future information
    leakage_cols = ["reservation_status", "reservation_status_date"]
    cols_to_drop = [col for col in leakage_cols if col in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    # company ID -> binary flag
    if "company" in X.columns:
        X["company_booking"] = (X["company"] != 0).astype(int)

    # agent ID -> binary flag
    if "agent" in X.columns:
        X["has_agent"] = (X["agent"] != 0).astype(int)
        X = X.drop(columns="agent")

    # parking spaces -> simpler yes/no feature
    if "required_car_parking_spaces" in X.columns:
        X["has_parking"] = (X["required_car_parking_spaces"] > 0).astype(int)
        X = X.drop(columns="required_car_parking_spaces")

    return X

def build_numerical_preprocessor(numeric_features, binary_features):
    """
    Create a ColumnTransformer that preprocesses numerical features.

    Parameters
    ----------
    numeric_features : list
        Columns that should be scaled.

    binary_features : list
        Binary indicator columns (0/1) that should only be imputed.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer for numerical columns.
    """

    # Continuous numerical features
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fill missing values robustly
        ("scaler", RobustScaler())  # scale while being robust to outliers
    ])

    # Binary features
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
        # no scaling because 0/1 values are already normalized
    ])

    column_preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("bin", binary_pipeline, binary_features),
    ])

    return column_preprocessor

def build_preprocessing_pipeline(numeric_features, binary_features):
    """
    Build the full preprocessing pipeline including
    feature engineering and numerical preprocessing.
    """

    column_preprocessor = build_numerical_preprocessor(
        numeric_features,
        binary_features
    )

    preproc_pipeline = Pipeline([
        ("feature_engineering",
         FunctionTransformer(engineer_numerical_features, validate=False)),
        ("preprocessing", column_preprocessor)
    ])

    return preproc_pipeline
