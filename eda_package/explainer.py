"""
NoShowShield — SHAP explainability module.

Responsibilities:
    - Compute SHAP values for the XGBoost classifier
    - Global feature importance (which features matter most overall)
    - Local explanations (why THIS specific date/room has high risk)
    - Format explanations for the dashboard API response
"""
import shap
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .model import ModelManager
from .registry import WORKING_MODEL_FILE_NAME


class SHAPExplainer:
    """SHAP helper for model explanation and importance."""

    def __init__(
        self,
        model: XGBClassifier = None,
        feature_names: list = None,
        model_file: str = None,
    ):
        """Initialize SHAP explainer.

        Args:
            model (XGBClassifier, optional): Pre-loaded XGBoost model.
            feature_names (list, optional): List of column names used by the model.
            model_file (str, optional): Path/name of model file to load via ModelManager.
                If not provided and no model is passed, uses WORKING_MODEL_FILE_NAME.
        """
        if model is None:
            model_file = model_file or WORKING_MODEL_FILE_NAME
            model_manager = ModelManager(file_name=model_file)
            self.model = model_manager.load()
        else:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided or loaded for SHAP explanation")

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            parsed_names = None
            if hasattr(self.model, "get_booster"):
                try:
                    parsed_names = self.model.get_booster().feature_names
                except Exception:
                    parsed_names = None
            if parsed_names is not None:
                self.feature_names = parsed_names
            else:
                raise ValueError(
                    "feature_names must be provided if it cannot be inferred from the model"
                )

        # SHAP may require the raw booster to avoid model-internal format mismatch
        self.explainer = None
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            if hasattr(self.model, "get_booster"):
                try:
                    self.explainer = shap.TreeExplainer(
                        self.model.get_booster(), feature_names=self.feature_names
                    )
                except Exception as e2:
                    raise ValueError(
                        "Cannot initialize SHAP explainer from model or booster: "
                        f"({e}) / ({e2})"
                    )
            else:
                raise ValueError(
                    "Cannot initialize SHAP explainer from model: " f"{e}"
                )

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the given dataset."""
        if self.explainer is None:
            raise ValueError("SHAP explainer is not initialized")

        shap_vals = self.explainer.shap_values(X)
        # shap.TreeExplainer returns list for multiclass; select binary/mode
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        return np.array(shap_vals)

    def global_feature_importance(
        self,
        X: pd.DataFrame = None,
        shap_values: np.ndarray = None,
        top_n: int = None,
    ) -> pd.DataFrame:
        """Compute global feature importance by mean abs SHAP values."""
        if shap_values is None:
            if X is None:
                raise ValueError("X or shap_values must be provided")
            shap_values = self.compute_shap_values(X)

        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values(by="importance", ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df.reset_index(drop=True)

    def local_explanation(self, X: pd.DataFrame, index: int) -> pd.DataFrame:
        """Generate local explanation for one row in X."""
        shap_values = self.compute_shap_values(X)
        if index < 0 or index >= shap_values.shape[0]:
            raise IndexError("Index out of range for provided SHAP values")

        local_shap_values = shap_values[index]
        local_explanation_df = pd.DataFrame(
            {"feature": self.feature_names, "shap_value": local_shap_values}
        ).sort_values(by="shap_value", ascending=False)
        return local_explanation_df.reset_index(drop=True)

    def format_explanation_for_api(self, explanation_df: pd.DataFrame) -> dict:
        """Format explanation DataFrame for API response."""
        return explanation_df.to_dict(orient="records")

    @classmethod
    def from_working_model(cls, feature_names: list = None):
        """Convenience constructor to load working model from disk."""
        return cls(model=None, feature_names=feature_names, model_file=WORKING_MODEL_FILE_NAME)
