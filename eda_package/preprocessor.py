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
    df:pd.DataFrame,
    limit:int
    )-> pd.DataFrame:
    """
    This function takes in a dataframe and group countries with less then 'limit' entries in Other category
    The function adds a new column called 'country_group' and leaves the 'country' column as-is
    """

    country_counts = df['country'].value_counts()
    countries_included = country_counts[country_counts >= limit].index

    df['country_group'] = df['country'].apply(
        lambda x: x if x in countries_included else 'Other'
    )

    #df = df.drop(columns=['country'])

    return df

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from the raw hotel booking dataset:
    - removes the target if it is present
    - removes known leakage columns
    - converts ID-like columns into binary indicators
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
        X["has_company"] = (X["company"] != 0).astype(int)
        X = X.drop(columns="company")

    # agent ID -> binary flag
    if "agent" in X.columns:
        X["has_agent"] = (X["agent"] != 0).astype(int)
        X = X.drop(columns="agent")

    # parking spaces -> simpler yes/no feature
    if "required_car_parking_spaces" in X.columns:
        X["has_parking"] = (X["required_car_parking_spaces"] > 0).astype(int)
        X = X.drop(columns="required_car_parking_spaces")

    return X

def get_feature_lists(df: pd.DataFrame):
    """
    Automatically determine numeric and binary feature lists
    based on dataframe dtypes and values
    """
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Identify binary columns
    binary_features = []
    for col in numeric_cols:
        unique_vals = set(df[col].dropna().unique())

        # Check if column only contains 0 and 1
        if unique_vals.issubset({0, 1}):
            binary_features.append(col)

    # Remaining numeric = true numeric
    numeric_features = [
        col for col in numeric_cols if col not in binary_features
    ]

    return numeric_features, binary_features

def create_preprocessing_pipeline(X: pd.DataFrame):
    X = engineer_features(X)
    numerical_features, binary_features = get_feature_lists(X)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    column_preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("bin", binary_pipeline, binary_features)
    ])

    preproc_pipeline = Pipeline([
        ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
        ("preprocessing", column_preprocessor)
    ])

    return preproc_pipeline

#df = engineer_numerical_features(df)
#numeric_features, binary_features = get_feature_lists(df)
#preproc_pipeline = build_preprocessing_pipeline(
   # numeric_features=numeric_features,
   # binary_features=binary_features)





def preproc_test():
    print("Hello from preprocessor!")
