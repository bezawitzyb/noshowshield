"""
NoShowShield — Model training and inference.

Responsibilities:
    - Build the full sklearn Pipeline (preprocessor + model)
    - Train the model with cross-validation
    - Predict cancellation probabilities
    - Evaluate model performance (AUC, precision, recall, calibration)

Usage:
    from noshowshield.eda_package.model import build_pipeline, train_model, evaluate_model

    pipeline = build_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
"""
import os
import joblib
import pandas as pd
import numpy as np
import pickle
import numpy as np
from .registry import *
from pathlib import Path #for file handling
from eda_package import *
from eda_package.data import load_raw_data, clean_data, temporal_split_v2, temporal_split, split_X_y
from eda_package.features import engineer_features
#from eda_package.preprocessor import (
from .data import load_raw_data, clean_data, temporal_split_v2, temporal_split, split_X_y
from .features import engineer_features
from .preprocessor import (
    group_countries,
    get_feature_lists,
    create_preprocessor,
    fit_transform_preprocessor,
    transform_preprocessor,
    preprocess_pipeline
)
from eda_package.registry import ORDINAL_FEATURES_MAP, COUNTRY_LIMIT, WORKING_MODEL_FILE_NAME
from .registry import ORDINAL_FEATURES_MAP, COUNTRY_LIMIT
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "working_model.pkl"

class BookingPredictor():

    def __init__(self,
                 file_name: str = None,
                 X_train_processed: pd.DataFrame = None,
                 y_train: pd.Series = None):

        if X_train_processed is not None and y_train is not None:
            self.train_model(X_train_processed, y_train)
            #add: save model to file_name if provided
        else:
            self.load_model(file_name=file_name)

    def train_model(self, X_train_processed, y_train):

        parameters = {
            'n_estimators': 100, #100, 300
            'max_depth': 3, #3, 10
            'learning_rate': 0.2, #0.2, 0.05
            'gamma': 10, #10, 1
        #    'lambda': 1,
        #    'alpha': 0,
            'subsample': 0.5, #minimal impact
            'colsample_bytree': 0.3, #minimal impact
            'min_child_weight': 2, #minimal impact
            'random_state': 0,
            'scale_pos_weight': 3, #impact
            'eval_metric': 'logloss'
        }

        self.model = XGBClassifier(**parameters)
        self.model.fit(X_train_processed,y_train)

    def test(self, X_test_processed, y_test):

        print('Testing', type(self.model))

        y_predicted = self.model.predict(X_test_processed)

        print(f'Accuracy: {round(accuracy_score(y_test, y_predicted),2)}')
        print(f'Recall: {round(recall_score(y_test, y_predicted),2)}')
        print(f'Precision: {round(precision_score(y_test, y_predicted),2)}')
        print(f'F1 score: {round(f1_score(y_test, y_predicted),2)}')
        print(classification_report(y_test,y_predicted))

    def predict(self, X_processed):

        return self.model.predict(X_processed)

    def predict_proba(self, X_processed):

        return self.model.predict_proba(X_processed)

    def save_model(self, file_name: str = None):

        if file_name is None:
            file_name = WORKING_MODEL_FILE_NAME

        url = '../models/' + file_name
        #print('Saving: ', url)
        pickle.dump(self.model, open(url, 'wb'))

    def load_model(self, file_name: str = None):

        if file_name is None:
            file_name = WORKING_MODEL_FILE_NAME

#        self.model = XGBClassifier()
        # url = '../models/' + file_name
        # #print('Loading: ', url)
        # #self.model = pickle.load(open(url, 'rb'))
        model_path = Path(__file__).resolve().parent.parent / "models" / file_name
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)



def the_brain():
    df = load_raw_data()
    df = clean_data(df)

    #SHOULD WE MOVE THIS TO FEATURES OR DATA?
    df = group_countries(df, COUNTRY_LIMIT)

    df = engineer_features(df)

    training_set, test_set = temporal_split(df, '2017-03-01')

    X_train, y_train= split_X_y(training_set)
    X_test, y_test= split_X_y(test_set)

    # 4. Feature lists (train only)
    feature_lists = get_feature_lists(X_train, ORDINAL_FEATURES_MAP)

    # 5. Preprocessor
    preprocessor = create_preprocessor(feature_lists, ORDINAL_FEATURES_MAP)

    # 6. Transform
    X_train_processed = fit_transform_preprocessor(X_train, preprocessor)
    X_test_processed = transform_preprocessor(X_test, preprocessor)


    model = LogisticRegression()
    model.fit(X_train_processed, y_train)
    y_predicted = model.predict(X_test_processed)

    print(f'Accuracy: {round(accuracy_score(y_test, y_predicted),2)}')
    print(f'Recall: {round(recall_score(y_test, y_predicted),2)}')
    print(f'Precision: {round(precision_score(y_test, y_predicted),2)}')
    print(f'F1 score: {round(f1_score(y_test, y_predicted),2)}')


def train_model():
    df = load_raw_data()
    df = clean_data(df)
    df = group_countries(df, COUNTRY_LIMIT)
    df = engineer_features(df)

    training_set, test_set = temporal_split(df, "2017-03-01")

    X_train, y_train = split_X_y(training_set)
    X_test, y_test = split_X_y(test_set)

    feature_lists = get_feature_lists(X_train, ORDINAL_FEATURES_MAP)
    preprocessor = create_preprocessor(feature_lists, ORDINAL_FEATURES_MAP)

    X_train_processed = fit_transform_preprocessor(X_train, preprocessor)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_processed, y_train)

    return model, preprocessor


class SimpleModelPipeline:
    """
    End-to-end pipeline:
        1.  Load → clean → feature-engineer → temporal split
        2.  Train a calibrated logistic regression
        3.  Predict per-booking cancel probability
        4.  Aggregate by (date, room_type): expected cancellations,
            std dev, 95 % CI
        5.  For each group, sweep extra bookings 0 … 30 using the
            Poisson-Binomial distribution and pick the level that
            maximises  E[revenue] – E[relocation cost]  while keeping
            P(relocation) ≤ max_risk
        6.  Return a single DataFrame with everything
    """

    def __init__(
        self,
        path: str = "/Users/beza/code/bezawitzyb/noshowshield/raw_data/hotel_bookings.csv",
        country_limit: int = COUNTRY_LIMIT,
        split_year: int = SPLIT_YEAR,
        ordinal_features_map: dict = None,
        model_folder: str = "models",
        random_state: int = 42,
        relocation_cost: float = DEFAULT_RELOCATION_COST,
        max_risk: float = DEFAULT_MAX_RISK,
        max_extra_sweep: int = MAX_EXTRA_SWEEP,
    ):
        self.path = path
        self.country_limit = country_limit
        self.split_year = split_year
        self.ordinal_features_map = ordinal_features_map or ORDINAL_FEATURES_MAP
        self.model_folder = model_folder
        self.random_state = random_state
        self.relocation_cost = relocation_cost
        self.max_risk = max_risk
        self.max_extra_sweep = max_extra_sweep

        self.model = None
        self.test_df = None       # raw test rows (kept for grouping)
        self.capacity_map = None  # inferred {room_type: capacity}

    # ================================================================
    #  1.  DATA PIPELINE
    # ================================================================
    def _build_arrival_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compose a datetime from the split year / month-name / day columns."""
        df = df.copy()
        df["arrival_date"] = pd.to_datetime(
            df["arrival_date_year"].astype(str)
            + "-"
            + df["arrival_date_month"]
            + "-"
            + df["arrival_date_day_of_month"].astype(str),
            format="%Y-%B-%d",
        )
        return df

    def _infer_capacity(self, df: pd.DataFrame) -> dict:
        """
        Count only bookings that actually showed up (is_canceled == 0).
        The max show-ups on any date is a tight lower bound on true capacity.
        """
        showed_up = df[df["is_canceled"] == 0]

        counts = (
            showed_up
            .groupby(["arrival_date", "assigned_room_type"])
            .size()
            .reset_index(name="n_showups")
        )
        capacity = (
            counts.groupby("assigned_room_type")["n_showups"]
            .max()
            .to_dict()
        )
        return capacity

    def load_and_preprocess(self):
        """Run the full data pipeline and return processed train/test arrays."""
        df = load_raw_data(self.path)
        df = clean_data(df)
        df = group_countries(df, self.country_limit)
        df = engineer_features(df)
        df = self._build_arrival_date(df)

        # Infer capacity from the FULL dataset (before splitting)
        self.capacity_map = self._infer_capacity(df)

        train, test = temporal_split(df, self.split_year)
        self.test_df = test.copy()

        X_train, y_train = split_X_y(train)
        X_test, y_test = split_X_y(test)

        # Drop columns that are only used for aggregation, not modelling
        drop_cols = ["arrival_date"]
        X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

        X_train_processed, X_test_processed, _ = preprocess_pipeline(
            X_train, X_test, self.ordinal_features_map
        )
        return X_train_processed, X_test_processed, y_train, y_test

    # ================================================================
    #  2.  MODEL TRAINING
    # ================================================================
    def train(self, X_train, y_train):
        base = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.model.fit(X_train, y_train)
        return self.model

    # ================================================================
    #  3.  PREDICTION
    # ================================================================
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)[:, 1]

    # ================================================================
    #  4.  EVALUATION
    # ================================================================
    def evaluate(self, y_true, y_pred, y_proba, verbose=True):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred, pos_label=1),
            "recall": recall_score(y_true, y_pred, pos_label=1),
            "brier": brier_score_loss(y_true, y_proba),
        }
        targets = {"roc_auc": 0.85, "precision": 0.80, "recall": 0.75, "brier": 0.15}

        if verbose:
            print("\n── Evaluation ──")
            for k, v in metrics.items():
                line = f"  {k:<12}: {v:.4f}"
                if k in targets:
                    target = targets[k]
                    ok = (v >= target) if k != "brier" else (v <= target)
                    line += f"  (target {'≤' if k == 'brier' else '≥'} {target}  {'✅' if ok else '❌'})"
                print(line)
        return metrics

    # ================================================================
    #  5.  POISSON-BINOMIAL HELPERS
    # ================================================================
    @staticmethod
    def _poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
        """
        Exact PMF via dynamic programming.
        probs : array of individual *cancellation* probabilities.
        Returns P(k cancellations) for k = 0 … n.
        """
        probs = np.asarray(probs, dtype=np.float64)
        n = len(probs)
        pmf = np.zeros(n + 1)
        pmf[0] = 1.0
        for p in probs:
            new = np.empty_like(pmf)
            new[0] = pmf[0] * (1 - p)
            new[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
            pmf = new
        return pmf

    @staticmethod
    def _poisson_binomial_stats(probs: np.ndarray):
        """Mean and variance of total cancellations."""
        probs = np.asarray(probs, dtype=np.float64)
        mean = probs.sum()
        var = (probs * (1 - probs)).sum()
        return mean, var

    # ================================================================
    #  6.  OVERBOOKING OPTIMISER  (per group)
    # ================================================================
    def _optimise_group(
        self,
        cancel_probs: np.ndarray,
        capacity: int,
        mean_adr: float,
    ) -> dict:
        """
        Sweep extra bookings 0 … max_extra_sweep.
        For each candidate level, compute:
            • P(show-ups > capacity)  via Poisson-Binomial on show-up probs
            • expected revenue gain   = extra × mean_adr
            • expected relocation cost = P(relocate) × E[excess] × relocation_cost
            • net profit              = revenue − cost

        Stop when P(relocate) > max_risk.
        Return the best (extra, profit) and supporting numbers.
        """
        cancel_probs = np.asarray(cancel_probs, dtype=np.float64)
        n_current = len(cancel_probs)
        show_probs = 1.0 - cancel_probs          # per-booking show-up probability

        best = {
            "recommended_extra": 0,
            "recommended_total": n_current,
            "net_benefit": 0.0,
            "additional_revenue": 0.0,
            "expected_relocation_cost": 0.0,
            "relocation_probability": 0.0,
        }

        for extra in range(0, self.max_extra_sweep + 1):
            total = n_current + extra

            # For the extra bookings we don't have individual probs,
            # so we use the group's mean cancel probability as the best
            # available estimate for hypothetical new bookings.
            if extra > 0:
                mean_cancel = cancel_probs.mean()
                extended_show = np.concatenate([
                    show_probs,
                    np.full(extra, 1.0 - mean_cancel),
                ])
            else:
                extended_show = show_probs

            # PMF of total show-ups (k = 0 … total)
            show_pmf = self._poisson_binomial_pmf(extended_show)

            # P(need to relocate) = P(show-ups > capacity)
            if capacity + 1 <= total:
                p_relocate = show_pmf[capacity + 1:].sum()
            else:
                p_relocate = 0.0

            # If we've breached the risk ceiling, stop searching
            if p_relocate > self.max_risk:
                break

            # E[excess guests] = Σ (k - capacity) × P(k)  for k > capacity
            expected_excess = sum(
                (k - capacity) * show_pmf[k]
                for k in range(capacity + 1, total + 1)
            )

            revenue = extra * mean_adr
            cost = expected_excess * self.relocation_cost
            profit = revenue - cost

            if profit >= best["net_benefit"]:
                best = {
                    "recommended_extra": extra,
                    "recommended_total": total,
                    "net_benefit": round(profit, 2),
                    "additional_revenue": round(revenue, 2),
                    "expected_relocation_cost": round(cost, 2),
                    "relocation_probability": round(p_relocate, 4),
                }

        return best

    # ================================================================
    #  7.  AGGREGATE + RECOMMEND  (all groups)
    # ================================================================
    def aggregate_and_recommend(
        self,
        raw_df: pd.DataFrame,
        X_processed,
        group_cols: tuple = ("arrival_date", "assigned_room_type"),
    ) -> pd.DataFrame:
        """
        1. Attach cancel probs and ADR to raw test rows.
        2. Group by (date, room_type).
        3. For each group compute:
             - total_bookings, expected_cancellations, std, 95% CI
             - expected_show_ups
             - capacity (inferred)
             - optimal overbooking recommendation + financials
        4. Return a single, flat DataFrame.
        """
        probs = self.predict_proba(X_processed)
        df = raw_df.copy()
        df["cancel_prob"] = probs

        # ── group-level aggregation ──
        grouped = (
            df.groupby(list(group_cols))
            .agg(
                total_bookings=("cancel_prob", "size"),
                expected_cancellations=("cancel_prob", "sum"),
                cancel_prob_mean=("cancel_prob", "mean"),
                cancel_prob_std=("cancel_prob", "std"),
                mean_adr=("adr", "mean"),
                individual_probs=("cancel_prob", list),
            )
            .reset_index()
        )

        # ── Poisson-Binomial stats per group ──
        grouped["variance"] = grouped["individual_probs"].apply(
            lambda ps: np.sum(np.array(ps) * (1 - np.array(ps)))
        )
        grouped["std_cancellations"] = np.sqrt(grouped["variance"])

        grouped["ci_lower"] = (
            grouped["expected_cancellations"] - 1.96 * grouped["std_cancellations"]
        ).clip(lower=0)
        grouped["ci_upper"] = (
            grouped["expected_cancellations"] + 1.96 * grouped["std_cancellations"]
        )

        grouped["expected_show_ups"] = (
            grouped["total_bookings"] - grouped["expected_cancellations"]
        )

        # ── map inferred capacity ──
        room_col = "assigned_room_type"
        grouped["capacity"] = grouped[room_col].map(self.capacity_map)

        # Fallback: if a room type wasn't seen, use total_bookings
        grouped["capacity"] = grouped["capacity"].fillna(grouped["total_bookings"])
        grouped["capacity"] = grouped["capacity"].astype(int)

        # ── run the optimiser for every group ──
        recommendations = []
        for _, row in grouped.iterrows():
            rec = self._optimise_group(
                cancel_probs=np.array(row["individual_probs"]),
                capacity=row["capacity"],
                mean_adr=row["mean_adr"],
            )
            recommendations.append(rec)

        rec_df = pd.DataFrame(recommendations)
        result = pd.concat(
            [grouped.drop(columns=["individual_probs"]).reset_index(drop=True), rec_df],
            axis=1,
        )

        # ── tidy column order ──
        leading = list(group_cols) + [
            "capacity",
            "total_bookings",
            "expected_cancellations",
            "std_cancellations",
            "ci_lower",
            "ci_upper",
            "expected_show_ups",
            "mean_adr",
            "recommended_extra",
            "recommended_total",
            "additional_revenue",
            "expected_relocation_cost",
            "net_benefit",
            "relocation_probability",
        ]
        extra_cols = [c for c in result.columns if c not in leading]
        result = result[[c for c in leading if c in result.columns] + extra_cols]

        return result

    # ================================================================
    #  8.  SAVE / LOAD
    # ================================================================
    def save_model(self, model_name="model", add_timestamp=True):
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(self.model_folder, exist_ok=True)
        ts = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if add_timestamp else ""
        path = os.path.join(self.model_folder, f"{model_name}{ts}.joblib")
        joblib.dump(self.model, path)
        print(f"Model saved → {path}")
        return path

    def load_model(self, path):
        self.model = joblib.load(path)
        return self.model

    # ================================================================
    #  9.  FULL PIPELINE  (one-call entry point)
    # ================================================================
    def run(self) -> dict:
        """
        Execute the entire pipeline and return:
            - model           : the trained classifier
            - metrics         : dict of evaluation scores
            - recommendations : DataFrame with group stats +
                                overbooking recommendation + financials
        """
        # Data
        X_train, X_test, y_train, y_test = self.load_and_preprocess()

        # Train
        self.train(X_train, y_train)

        # Evaluate
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        metrics = self.evaluate(y_test, y_pred, y_proba)

        # Aggregate + optimise
        recommendations = self.aggregate_and_recommend(self.test_df, X_test)

        return {
            "model": self.model,
            "metrics": metrics,
            "recommendations": recommendations,
        }

    def run_from_saved_model(self, model_path: str) -> dict:
        """
        Execute the pipeline using a pre-trained saved model.
        Skips training — loads model from disk, then evaluates
        and generates overbooking recommendations.

        Parameters
        ----------
        model_path : str
            Path to a .joblib model file saved via self.save_model().

        Returns
        -------
        dict with keys:
            - model           : the loaded classifier
            - metrics         : dict of evaluation scores
            - recommendations : DataFrame with group stats +
                                overbooking recommendations + financials
        """
        # Data
        X_train, X_test, y_train, y_test = self.load_and_preprocess()

        # Load instead of train
        self.load_model(model_path)
        print(f"Model loaded from → {model_path}")

        # Evaluate
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        metrics = self.evaluate(y_test, y_pred, y_proba)

        # Aggregate + optimise
        recommendations = self.aggregate_and_recommend(self.test_df, X_test)

        return {
            "model": self.model,
            "metrics": metrics,
            "recommendations": recommendations,
        }

    def get_recommendations(self, recs, dates, room_types):
        if isinstance(dates, str):
            dates = [dates]
        if isinstance(room_types, str):
            room_types = [room_types]

        timestamps = [pd.Timestamp(d) for d in dates]

        filtered = recs[
            (recs["arrival_date"].isin(timestamps)) &
            (recs["assigned_room_type"].isin(room_types))
        ].reset_index(drop=True)

        return filtered
