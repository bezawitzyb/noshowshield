"""
NoShowShield — Data loading and preparation

Responsibilities:
    - Load raw CSV
    - Clean raw data
    - Group rare countries into 'Other'
    - Split data with train_test_split
    - Separate features and target
    - Export one row as API-ready JSON payload

Usage:
    from eda_package.registry import *
    from eda_package.data import DataManager

    data_manager = DataManager()
    df = data_manager.prepare_dataset()
    X_train, X_test, y_train, y_test = data_manager.prepare_train_test_data()

"""

from pathlib import Path
from typing import Optional, Tuple
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .registry import LEAKY_COLS, COUNTRY_LIMIT


class DataManager:
    """
    Central class for loading, cleaning, preparing, and splitting hotel booking data.
    """

    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        country_limit: int = COUNTRY_LIMIT
    ):
        if raw_data_path is None:
            self.raw_data_path = (
                Path(__file__).resolve().parents[1]
                / "raw_data"
                / "hotel_bookings.csv"
            )
        else:
            self.raw_data_path = Path(raw_data_path)

        self.country_limit = country_limit
        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None

    def load_raw_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the raw hotel bookings CSV.
        Cache it in memory unless force_reload=True.
        """
        if self._raw_df is None or force_reload:
            self._raw_df = pd.read_csv(self.raw_data_path)

        return self._raw_df.copy()

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the dataset.

        Steps:
        1. Remove duplicates
        2. Convert reservation_status_date to datetime
        3. Fill missing values in agent, children, and company with 0
        4. Fill missing values in country with 'Other'
        """
        if df is None:
            if self._clean_df is not None:
                return self._clean_df.copy()
            df = self.load_raw_data()

        cleaned_df = df.copy()

        cleaned_df = cleaned_df.drop_duplicates()
        cleaned_df["reservation_status_date"] = pd.to_datetime(
            cleaned_df["reservation_status_date"]
        )
        cleaned_df["agent"] = cleaned_df["agent"].fillna(0)
        cleaned_df["children"] = cleaned_df["children"].fillna(0)
        cleaned_df["company"] = cleaned_df["company"].fillna(0)
        cleaned_df["country"] = cleaned_df["country"].fillna("Other")

        self._clean_df = cleaned_df.copy()

        return cleaned_df

    def group_countries(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Group countries with fewer than `limit` entries into 'Other'.

        Adds a new column called 'country_group'
        and leaves the original 'country' column unchanged.
        """
        if limit is None:
            limit = self.country_limit

        grouped_df = df.copy()

        country_counts = grouped_df["country"].value_counts()
        countries_included = country_counts[country_counts >= limit].index

        grouped_df["country_group"] = grouped_df["country"].apply(
            lambda x: x if x in countries_included else "Other"
        )

        return grouped_df

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "is_canceled",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataframe into train/test sets using train_test_split.
        """
        X = df.drop(columns=target_col)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        return X_train, X_test, y_train, y_test

    def prepare_clean_data(self) -> pd.DataFrame:
        """
        Load and clean the raw dataset.
        """
        df = self.load_raw_data()
        df = self.clean_data(df)
        return df

    def prepare_dataset(self) -> pd.DataFrame:
        """
        Final convenience method for the data class.

        Runs:
        1. load raw data
        2. clean data
        3. group countries
        """
        df = self.prepare_clean_data()
        df = self.group_countries(df)
        return df

    def prepare_train_test_data(
        self,
        target_col: str = "is_canceled",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Full end-to-end data preparation for modeling.

        Runs:
        1. load raw data
        2. clean data
        3. group countries
        4. split into train/test
        """
        df = self.prepare_dataset()

        X_train, X_test, y_train, y_test = self.split_data(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        return X_train, X_test, y_train, y_test

    def row_to_api_json(self, row_index: int = 0) -> str:
        """
        Convert one row into a JSON payload suitable for the FastAPI /predict endpoint.
        """
        df = self.prepare_dataset()

        if row_index >= len(df):
            raise IndexError("row_index is out of range.")

        row = df.iloc[row_index].copy()
        row = row.drop(labels=LEAKY_COLS, errors="ignore")

        data = row.to_dict()
        data = {
            key: (None if pd.isna(value) else value)
            for key, value in data.items()
        }

        return json.dumps(data, default=str)
