"""
NoShowShield — Model training and inference

Responsibilities:
    - Build the prediction model
    - Train on preprocessed training data
    - Predict classes and probabilities
    - Evaluate model performance
    - Save and load trained models

Usage:
    from eda_package.model import ModelManager

    model_manager = ModelManager()

    # Training
    model_manager.train(X_train_processed, y_train)
    metrics = model_manager.evaluate(X_test_processed, y_test)
    model_manager.save()

    # Inference / API
    model_manager.load()
    y_pred = model_manager.predict(X_pred_processed)
    y_prob = model_manager.predict_proba(X_pred_processed)
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

from .registry import WORKING_MODEL_FILE_NAME


class ModelManager:
    """
    Central class for model training, inference, evaluation, and persistence.
    """

    def __init__(
        self,
        file_name: str = WORKING_MODEL_FILE_NAME,
        model=None,
        model_params: Optional[Dict] = None
    ):
        self.path = Path(__file__).resolve().parent.parent / "models" / file_name
        self.model = model

        self.model_params = model_params or {
            'objective' : 'binary:logistic',
            'booster' : 'gbtree',
            'n_estimators': 500,
            'max_depth': 12,
            'learning_rate': 0.03,
            'gamma': 1,
            'subsample': 0.5, #minimal impact
            'colsample_bytree': 0.5, #minimal impact
            'min_child_weight': 3, #minimal impact
            'random_state': 0,
            'scale_pos_weight': 3, #impact
            'eval_metric': 'logloss'
        }

    def build_model(self):
        """
        Create the default XGBoost classification model.
        """
        self.model = XGBClassifier(**self.model_params)
        return self.model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model on preprocessed training data.
        """
        if self.model is None:
            self.build_model()

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict classes for preprocessed input data.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities for preprocessed input data.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model on preprocessed test data.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 2),
            "recall": round(recall_score(y_test, y_pred), 2),
            "precision": round(precision_score(y_test, y_pred), 2),
            "f1": round(f1_score(y_test, y_pred), 2),
            "auc": round(roc_auc_score(y_test, y_prob), 2),
        }

        return metrics

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """
        Final convenience method:
        1. build model if needed
        2. train on X_train / y_train
        3. evaluate on X_test / y_test
        """
        self.train(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)

        return self.model, metrics

    def save(self):
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.path)

    def load(self):
        """
        Load a trained model from disk.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"No model found at {self.path}")

        self.model = joblib.load(self.path)
        return self.model
