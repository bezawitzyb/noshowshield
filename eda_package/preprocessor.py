"""
NoShowShield — Sklearn preprocessing pipeline

Responsibilities:
    - Detect feature types
    - Build a ColumnTransformer
    - Fit on training data only
    - Transform train/test/new data consistently
    - Save and load fitted preprocessors

Usage:
    from eda_package.preprocessor import PreprocessorManager

    preprocessor_manager = PreprocessorManager()

    # Training
    X_train_processed, X_test_processed, preprocessor = (
        preprocessor_manager.prepare_train_test(X_train, X_test)
    )
    preprocessor_manager.save()

    # Later for inference / API
    preprocessor_manager.load()
    X_pred_processed = preprocessor_manager.transform(X_pred)
"""

from pathlib import Path
from typing import Optional, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

from .registry import ORDINAL_FEATURES_MAP, RELEVANT_FEATURES


class PreprocessorManager:
    """
    Central class for preprocessing tabular model inputs.
    """

    def __init__(
        self,
        ordinal_feature_map: Optional[Dict] = None,
        file_name: str = "preprocessor.joblib"
    ):
        self.ordinal_feature_map = (
            ordinal_feature_map if ordinal_feature_map is not None else ORDINAL_FEATURES_MAP
        )
        self.path = Path(__file__).resolve().parent.parent / "models" / file_name
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_lists: Optional[Dict] = None

    def get_feature_lists(self, X: pd.DataFrame) -> Dict:
        """
        Detect numerical, binary, one-hot categorical, and ordinal features.
        """
        X = X.copy()

        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numeric_candidates = X.select_dtypes(include=["number"]).columns.tolist()

        binary_features = []
        numerical_features = []

        for col in numeric_candidates:
            unique_vals = set(X[col].dropna().unique())

            if unique_vals.issubset({0, 1}) and len(unique_vals) <= 2:
                binary_features.append(col)
            else:
                numerical_features.append(col)

        ordinal_features = [
            col for col in categorical_features if col in self.ordinal_feature_map
        ]

        onehot_features = [
            col for col in categorical_features if col not in self.ordinal_feature_map
        ]

        feature_lists = {
            "numerical_features": numerical_features,
            "binary_features": binary_features,
            "onehot_features": onehot_features,
            "ordinal_features": ordinal_features,
        }

        self.feature_lists = feature_lists
        return feature_lists

    def create_preprocessor(
        self,
        feature_lists: Optional[Dict] = None
    ) -> ColumnTransformer:
        """
        Create an unfitted ColumnTransformer.
        """
        if feature_lists is None:
            if self.feature_lists is None:
                raise ValueError("feature_lists not provided and not yet computed.")
            feature_lists = self.feature_lists

        numerical_features = feature_lists["numerical_features"]
        binary_features = feature_lists["binary_features"]
        onehot_features = feature_lists["onehot_features"]
        ordinal_features = feature_lists["ordinal_features"]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        binary_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])

        onehot_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformers = [
            ("num", numeric_pipeline, numerical_features),
            ("bin", binary_pipeline, binary_features),
            ("cat_onehot", onehot_pipeline, onehot_features),
        ]

        if ordinal_features:
            ordinal_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[
                            self.ordinal_feature_map[col]
                            for col in ordinal_features
                        ]
                    )
                )
            ])
            transformers.append(("cat_ordinal", ordinal_pipeline, ordinal_features))

        self.preprocessor = ColumnTransformer(transformers=transformers)
        return self.preprocessor

    def fit(self, X_train: pd.DataFrame):
        """
        Fit the preprocessor on training data only.
        """
        self.get_feature_lists(X_train)
        self.create_preprocessor()
        self.preprocessor.fit(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a dataframe using the fitted preprocessor.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted or loaded.")

        X_processed = self.preprocessor.transform(X)
        feature_names = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X.index
        )

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and transform it.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def filter_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ):
        features_to_include = RELEVANT_FEATURES
        features_to_drop = []
        print('filtering: ', X_train.shape)

        for column in X_train:
            if column not in features_to_include:
                features_to_drop.append(column)

        X_train = X_train.drop(columns=features_to_drop)
        X_test = X_test.drop(columns=features_to_drop)

        print('filtered: ', X_train.shape)

        return X_train, X_test

    def prepare_train_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        filter_freatures: bool = False
    ):
        """
        Final convenience method:
        1. detect feature types from X_train
        2. build preprocessor
        3. fit on X_train
        4. transform X_train
        5. transform X_test
        Returns:
            X_train_processed, X_test_processed, preprocessor
        """
        X_train_processed = self.fit_transform(X_train)
        X_test_processed = self.transform(X_test)

        if self.filter_features:
            X_train_processed, X_test_processed = self.filter_features(X_train_processed, X_test_processed)

        return X_train_processed, X_test_processed, self.preprocessor

    def save(self):
        """
        Save the fitted preprocessor to disk.
        """
        if self.preprocessor is None:
            raise ValueError("No fitted preprocessor to save.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, self.path)

    def load(self):
        """
        Load a fitted preprocessor from disk.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"No preprocessor found at {self.path}")

        self.preprocessor = joblib.load(self.path)
        return self.preprocessor
