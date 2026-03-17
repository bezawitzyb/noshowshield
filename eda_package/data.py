"""
NoShowShield — Data loading and cleaning

Responsibilities:
    - Load raw CSV from Kaggle
    - Drop duplicate columns and handle missing values
    - Save/load processed files

Usage:
    from noshowshield.eda_package.data import load_raw_data, clean_data

    df = load_raw_data()
    df = clean_data(df)
    train, test = temporal_split(df)
"""

import pandas as pd
import numpy as np

def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Load the raw hotel bookings CSV.

    Returns:
        DataFrame with 119,390 rows × 32 columns (if unmodified Kaggle file).
    """

    if path is None:
        path = '../raw_data/hotel_bookings.csv'
    data = pd.read_csv(path)
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes in a dataframe and performs the following cleaning steps:
    1. Remove duplicates
    2. Convert reservation_status_date to datetime format
    3. Set missing values for agent, children and company to 0
    4. Impute rows with missing values in the country column with 'Other'
    """

    df = df.copy() # Create a copy of the original dataframe to avoid modifying it directly (to avoid warnings)
    #Remove duplicates
    df = df.drop_duplicates()
    #datetime format for reservation_status_date
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    #Set missing value for agent, children and company to 0
    df['agent'] = df['agent'].fillna(0)
    df['children'] = df['children'].fillna(0)
    df['company'] = df['company'].fillna(0)
    #Impute rows with missing values in the country column with 'Other'
    df['country'] = df['country'].fillna('Other')

    #Are there other data types that need to be changed or rows to be dropped?
    #Consider dropping some of the columns?

    return df

def temporal_split(df: pd.DataFrame)-> pd.DataFrame:
    """
    Split data temporally:

    This prevents data leakage from future bookings into the training set.
    Default: train on 2015–2016, test on 2017.

    Returns:
        train, test
    """


    pass
