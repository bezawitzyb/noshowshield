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

def group_countries(
    data:pd.DataFrame,
    limit:int
    )-> pd.DataFrame:
    """
    This function takes in a dataframe and group countries with less then 'limit' entries in Other category
    The function adds a new column called 'country_group' and leaves the 'country' column as-is
    """

    country_counts = data['country'].value_counts()
    countries_included = country_counts[country_counts >= limit].index

    data['country_group'] = data['country'].apply(
        lambda x: x if x in countries_included else 'Other'
    )

    #data = data.drop(columns='country')

    return data
