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
        X["company_booking"] = X["company"].notna().astype(int)
        X = X.drop(columns="company")

    # agent ID -> binary flag
    if "agent" in X.columns:
        X["has_agent"] = X["agent"].notna().astype(int)
        X = X.drop(columns="agent")

    # parking spaces -> simpler yes/no feature
    if "required_car_parking_spaces" in X.columns:
        X["has_parking"] = (X["required_car_parking_spaces"] > 0).astype(int)
        X = X.drop(columns="required_car_parking_spaces")

    # children / babies flags
    if "children" in X.columns:
        X["has_children"] = (X["children"].fillna(0) > 0).astype(int)

    if "babies" in X.columns:
        X["has_babies"] = (X["babies"].fillna(0) > 0).astype(int)

    # family booking flag
    if {"children", "babies"}.issubset(X.columns):
        X["is_family"] = (X[["children", "babies"]].fillna(0).sum(axis=1) > 0).astype(int)

    # total stay length
    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(X.columns):
        X["total_stay"] = (
            X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
        )

    # total guests
    if {"adults", "children", "babies"}.issubset(X.columns):
        X["total_guests"] = (X["adults"] + X["children"].fillna(0) + X["babies"].fillna(0))

    return X

# # turn this into a function , then another function to do both
# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", RobustScaler())
# ])

# binary_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent"))
# ])

# column_preprocessor = ColumnTransformer([
#     ("num", numeric_pipeline, numeric_features),
#     ("bin", binary_pipeline, binary_features),
# ])

# preproc_pipeline = Pipeline([
#     ("feature_engineering", FunctionTransformer(engineer_hotel_features, validate=False)),
#     ("preprocessing", column_preprocessor)
# ])
import pandas as pd
import numpy as np

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
