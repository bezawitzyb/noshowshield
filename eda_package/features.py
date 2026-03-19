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

        if drop_leakage:
            df = self.drop_leaky_columns(df)

        return df
