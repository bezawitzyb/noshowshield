# Info:
# I removed all features that used a yearly split logic,
# as we now split via traintestsplit,
# and as they dont improve performance metrics

"""
NoShowShield — Feature Engineering Module

Responsibilities:
    - Create derived features
    - Drop leakage columns for modeling
    - Provide one final method to run all feature engineering steps

Usage:
    from eda_package.features import FeatureEngineer

    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.engineer_features(X_train)
    X_test = feature_engineer.engineer_features(X_test)
"""
from typing import Optional, List
import pandas as pd
import numpy as np

from .registry import LEAKY_COLS


class FeatureEngineer:
    def __init__(self, leaky_cols: Optional[List[str]] = None):
        self.leaky_cols = leaky_cols if leaky_cols is not None else LEAKY_COLS

    def drop_leaky_columns(
        self,
        df: pd.DataFrame,
        extra_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = df.copy()

        cols_to_drop = list(self.leaky_cols)
        if extra_cols is not None:
            cols_to_drop.extend(extra_cols)

        return df.drop(columns=cols_to_drop, errors="ignore")

    def add_room_type_mismatch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["room_type_mismatch"] = (
            df["reserved_room_type"] != df["assigned_room_type"]
        ).astype(int)
        return df

    def add_special_requests_per_guest(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["special_requests_per_guest"] = (
            df["total_of_special_requests"] / df["adults"].replace(0, 1)
        )
        return df

    def add_weekend_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        total = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["weekend_ratio"] = np.where(
            total > 0,
            df["stays_in_weekend_nights"] / total,
            0.0,
        )
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        drop_leakage: bool = True
    ) -> pd.DataFrame:
        df = df.copy()

        df = self.add_room_type_mismatch(df)
        df = self.add_special_requests_per_guest(df)
        df = self.add_weekend_ratio(df)

#        df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

        if drop_leakage:
            df = self.drop_leaky_columns(df)

        return df
from .registry import SPLIT_YEAR


def add_room_type_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 when assigned room differs from reserved room.

    No temporal concern — purely row-level, no aggregation involved.
    """
    df['room_type_mismatch'] = (
        df['reserved_room_type'] != df['assigned_room_type']
    ).astype(int)
    return df


def add_adr_deviation_from_segment(
    df: pd.DataFrame,
    split_year: int = SPLIT_YEAR,
) -> pd.DataFrame:
    """
    How far a booking's ADR deviates from the average ADR
    in its market segment.

    Leakage prevention:
      Segment means are computed ONLY on rows where
      arrival_date_year < split_year, then mapped
      back to the full dataset. This way test-period
      bookings never influence the baseline they're
      measured against.
    """
    train_mask = df['arrival_date_year'] < split_year

    segment_means = (
        df.loc[train_mask]
        .groupby('market_segment')['adr']
        .mean()
    )

    global_mean = segment_means.mean()
    mapped_means = df['market_segment'].map(segment_means).fillna(global_mean)

    df['adr_deviation_from_segment'] = df['adr'] - mapped_means
    return df


def add_segment_cancel_rate(
    df: pd.DataFrame,
    split_year: int = SPLIT_YEAR,
) -> pd.DataFrame:
    """
    Historical cancellation rate for each market segment.

    Leakage prevention:
      Cancel rates are computed ONLY on rows where
      arrival_date_year < split_year. Without this,
      the target variable (is_canceled) from test-period
      bookings leaks directly into the feature — the model
      would be learning from future outcomes.
    """
    train_mask = df['arrival_date_year'] < split_year

    segment_rates = (
        df.loc[train_mask]
        .groupby('market_segment')['is_canceled']
        .mean()
    )

    global_rate = df.loc[train_mask, 'is_canceled'].mean()
    df['segment_cancel_rate'] = (
        df['market_segment'].map(segment_rates).fillna(global_rate)
    )
    return df


def add_special_requests_per_guest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of special requests to adult guests.

    No temporal concern — purely row-level arithmetic.
    """
    df['special_requests_per_guest'] = (
        df['total_of_special_requests'] / df['adults'].replace(0, 1)
    )
    return df


def add_weekend_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proportion of stay nights that fall on weekends.

    No temporal concern — purely row-level arithmetic.
    """
    total = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['weekend_ratio'] = np.where(
        total > 0,
        df['stays_in_weekend_nights'] / total,
        0.0,
    )
    return df


def engineer_features(
    df: pd.DataFrame,
    split_year: int = SPLIT_YEAR,
) -> pd.DataFrame:
    """
    Apply all feature engineering in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (train + test combined).
    split_year : int
        Rows with arrival_date_year < this value are treated
        as training data for any aggregation-based features.

    Returns
    -------
    pd.DataFrame with new columns added.
    """
    df = df.copy()
    df = add_room_type_mismatch(df)
    df = add_adr_deviation_from_segment(df, split_year)
#    df = add_segment_cancel_rate(df, split_year)
    df = add_special_requests_per_guest(df)
    df = add_weekend_ratio(df)

    return df

def engineer_features_v2(
    df: pd.DataFrame,
    split_year: int = SPLIT_YEAR,
) -> pd.DataFrame:
    """
    Apply all feature engineering in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (train + test combined).
    split_year : int
        Rows with arrival_date_year < this value are treated
        as training data for any aggregation-based features.

    Returns
    -------
    pd.DataFrame with new columns added.
    """
    df = df.copy()
    df = add_room_type_mismatch(df)
    df = add_adr_deviation_from_segment(df, split_year)
    #df = add_segment_cancel_rate(df, split_year)
    df = add_special_requests_per_guest(df)
    df = add_weekend_ratio(df)
    return df
