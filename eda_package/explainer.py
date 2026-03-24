import numpy as np
import pandas as pd
import xgboost as xgb


class ExplainerManager:
    """
    Handles SHAP explainability for the trained booking cancellation model.

    Responsibilities
    ----------------
    - Build an explainer from a trained XGBoost model
    - Aggregate transformed feature names into business-friendly groups
    - Produce local explanations for one booking
    - Produce global explanations across many bookings
    - Produce date-specific global explanations

    Implementation note
    -------------------
    SHAP values are computed via XGBoost's native pred_contribs=True rather
    than the shap library's TreeExplainer.  Both use the same exact tree-path
    algorithm; the native path avoids a version-incompatibility between the
    shap library and XGBoost 3.x that prevents TreeExplainer from being
    instantiated.
    """

    def __init__(self):
        self.booster = None

    def build_explainer(self, model_manager, X_background: pd.DataFrame = None):
        """
        Store the XGBoost booster for native SHAP computation.

        Parameters
        ----------
        model_manager : ModelManager
            Must contain the trained model in .model
        X_background : pd.DataFrame, optional
            Unused — kept for backwards-compatible call sites.
        """
        self.booster = model_manager.model.get_booster()
        return self.booster

    def _compute_shap_array(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute per-feature SHAP values using XGBoost's native pred_contribs.
        Reason: Error when using the TreeExplainer from the lecture.
        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            The last column returned by XGBoost (bias term) is dropped so
            the output aligns 1-to-1 with X.columns.
        """
        dmatrix = xgb.DMatrix(X)
        contribs = self.booster.predict(dmatrix, pred_contribs=True)
        return contribs[:, :-1]  # drop bias column

    def group_feature_name(self, name: str) -> str:
        """
        Collapse transformed feature names into user-friendly business feature groups.
        """
        for prefix in ["cat_onehot__", "cat_ordinal__", "cat__", "num__", "ord__", "bin__", "remainder__"]:
            if name.startswith(prefix):
                name = name[len(prefix):]

        keep_exact = {
            "lead_time",
            "arrival_date_year",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "days_in_waiting_list",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
            "special_requests_per_guest",
            "room_type_mismatch",
            "total_nights",
            "weekend_ratio",
            "agent",
            "company",
            "is_repeated_guest",
        }

        if name in keep_exact:
            return name

        grouped_prefixes = [
            "hotel_",
            "meal_",
            "country_",
            "country_group_",
            "market_segment_",
            "distribution_channel_",
            "reserved_room_type_",
            "assigned_room_type_",
            "deposit_type_",
            "customer_type_",
            "arrival_date_month_",
        ]

        for prefix in grouped_prefixes:
            if name.startswith(prefix):
                if prefix == "country_group_":
                    return "country"
                return prefix[:-1]

        return name

    def grouped_local_shap(
        self,
        shap_values_row: np.ndarray,
        feature_values_row: np.ndarray,
        feature_names,
    ) -> pd.DataFrame:
        """
        Aggregate local SHAP values for one booking into business-friendly groups.

        Parameters
        ----------
        shap_values_row : np.ndarray, shape (n_features,)
        feature_values_row : np.ndarray, shape (n_features,)
        feature_names : sequence of str
        """
        df = pd.DataFrame({
            "feature": list(feature_names),
            "feature_value": feature_values_row,
            "shap_value": shap_values_row,
        })

        df["feature_group"] = df["feature"].apply(self.group_feature_name)

        grouped = (
            df.groupby("feature_group", as_index=False)
            .agg({"shap_value": "sum"})
        )

        grouped["abs_shap"] = grouped["shap_value"].abs()
        grouped = grouped.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")

        return grouped

    def grouped_global_shap(
        self,
        shap_array: np.ndarray,
        feature_names,
    ) -> pd.DataFrame:
        """
        Aggregate global SHAP importance across many bookings into business-friendly groups.
        Uses mean absolute SHAP value.

        Parameters
        ----------
        shap_array : np.ndarray, shape (n_samples, n_features)
        feature_names : sequence of str
        """
        shap_df = pd.DataFrame(shap_array, columns=list(feature_names))
        mean_abs = shap_df.abs().mean(axis=0).reset_index()
        mean_abs.columns = ["feature", "mean_abs_shap"]

        mean_abs["feature_group"] = mean_abs["feature"].apply(self.group_feature_name)

        grouped = (
            mean_abs.groupby("feature_group", as_index=False)["mean_abs_shap"]
            .sum()
            .sort_values("mean_abs_shap", ascending=False)
        )

        return grouped

    def split_local_drivers(self, grouped_df: pd.DataFrame, top_n: int = 5):
        """
        Split grouped local SHAP into higher vs lower cancellation risk drivers.
        Assumes class 1 = canceled.
        """
        higher_risk = (
            grouped_df[grouped_df["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        lower_risk = (
            grouped_df[grouped_df["shap_value"] < 0]
            .sort_values("shap_value", ascending=True)
            .head(top_n)
            .reset_index(drop=True)
        )

        return higher_risk, lower_risk

    def transform_to_shap_df(
        self,
        X_processed,
        feature_names,
        index,
    ) -> pd.DataFrame:
        """
        Convert processed model input into a DataFrame with original transformed feature names.
        """
        try:
            return pd.DataFrame.sparse.from_spmatrix(
                X_processed,
                columns=feature_names,
                index=index
            )
        except Exception:
            return pd.DataFrame(
                X_processed,
                columns=feature_names,
                index=index
            )

    def explain_local(self, X_shap: pd.DataFrame, row_index=0):
        """
        Compute SHAP values for one or more rows and return grouped local explanation.
        """
        if self.booster is None:
            raise ValueError("Explainer not built yet. Call build_explainer() first.")

        shap_array = self._compute_shap_array(X_shap)
        grouped_local = self.grouped_local_shap(
            shap_values_row=shap_array[row_index],
            feature_values_row=X_shap.iloc[row_index].values,
            feature_names=X_shap.columns,
        )

        return {
            "shap_values": shap_array,
            "grouped_local_shap": grouped_local,
        }

    def explain_global(self, X_shap: pd.DataFrame):
        """
        Compute SHAP values across many rows and return grouped global importance.
        """
        if self.booster is None:
            raise ValueError("Explainer not built yet. Call build_explainer() first.")

        shap_array = self._compute_shap_array(X_shap)
        grouped_global = self.grouped_global_shap(shap_array, X_shap.columns)

        return {
            "shap_values": shap_array,
            "grouped_global_shap": grouped_global,
        }

    def explain_global_for_date(
        self,
        selected_date,
        X_raw: pd.DataFrame,
        data_manager,
        feature_engineer,
        preprocessor_manager,
        min_rows: int = 1,
    ):
        """
        Compute grouped global SHAP importance for all bookings on one selected arrival date.
        """
        if self.booster is None:
            raise ValueError("Explainer not built yet. Call build_explainer() first.")

        X = X_raw.copy()

        X["arrival_date"] = pd.to_datetime(
            X["arrival_date_day_of_month"].astype(str)
            + " "
            + X["arrival_date_month"].astype(str)
            + " "
            + X["arrival_date_year"].astype(str),
            format="%d %B %Y",
            errors="coerce"
        )

        selected_date = pd.Timestamp(selected_date)
        X_date = X[X["arrival_date"] == selected_date].copy()

        if len(X_date) == 0:
            return {
                "selected_date": selected_date,
                "n_bookings": 0,
                "grouped_global_shap": None,
                "X_date_raw": X_date,
                "X_date_shap": None,
                "shap_values_date": None,
                "message": f"No bookings found on {selected_date.date()} for the selected room type.",
            }

        X_date_model = X_date.drop(columns=["arrival_date"])

        if "country" in X_date_model.columns:
            X_date_model = data_manager.group_countries(X_date_model.copy())

        X_date_fe = feature_engineer.engineer_features(X_date_model.copy())
        X_date_processed = preprocessor_manager.transform(X_date_fe)

        feature_names = preprocessor_manager.preprocessor.get_feature_names_out()

        X_date_shap = self.transform_to_shap_df(
            X_processed=X_date_processed,
            feature_names=feature_names,
            index=X_date_model.index,
        )

        shap_array = self._compute_shap_array(X_date_shap)
        grouped_global_date = self.grouped_global_shap(shap_array, X_date_shap.columns)

        return {
            "selected_date": selected_date,
            "n_bookings": len(X_date),
            "grouped_global_shap": grouped_global_date,
            "X_date_raw": X_date,
            "X_date_shap": X_date_shap,
            "shap_values_date": shap_array,
            "message": None,
        }
