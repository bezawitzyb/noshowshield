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
import pickle
import numpy as np
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
    def __init__(
        self,
        path="/Users/beza/code/bezawitzyb/noshowshield/raw_data/hotel_bookings.csv",
        country_limit=COUNTRY_LIMIT,
        split_year=SPLIT_YEAR,
        ordinal_features_map=ORDINAL_FEATURES_MAP,
        model_folder="/Users/beza/code/bezawitzyb/noshowshield/models",
        random_state=42
    ):
        self.path = path
        self.country_limit = country_limit
        self.split_year = split_year
        self.ordinal_features_map = ordinal_features_map
        self.model_folder = model_folder
        self.random_state = random_state

        self.model = None

    # -----------------------------
    # Data Pipeline
    # -----------------------------
    def load_and_preprocess(self):
        df = load_raw_data(self.path)
        df = clean_data(df)
        df = group_countries(df, self.country_limit)
        df = engineer_features(df)

        train, test = temporal_split(df, self.split_year)
        X_train, y_train = split_X_y(train)
        X_test, y_test = split_X_y(test)

        X_train_processed, X_test_processed, preprocessor = preprocess_pipeline(
            X_train, X_test, self.ordinal_features_map
        )

        return X_train_processed, X_test_processed, y_train, y_test, preprocessor

    # -----------------------------
    # Model Training
    # -----------------------------
    def train(self, X_train, y_train):
        base_model = LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            random_state=self.random_state
        )

        self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        self.model.fit(X_train, y_train)

        return self.model

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)[:, 1]

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate(self, y_true, y_pred, y_proba, verbose=True):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred, pos_label=1),
            "recall": recall_score(y_true, y_pred, pos_label=1),
            "brier": brier_score_loss(y_true, y_proba)
        }

        targets = {
            "roc_auc": 0.85,
            "precision": 0.8,
            "recall": 0.75,
            "brier": 0.15
        }

        if verbose:
            print("\nMetrics:")
            for k, v in metrics.items():
                print(f"{k.capitalize():<10}: {v:.4f}", end="")
                if k in targets:
                    target = targets[k]
                    passed = (v >= target) if k != "brier" else (v <= target)
                    print(f" | Target: {target} | {'✅' if passed else '❌'}")
                else:
                    print()

        return metrics

    # -----------------------------
    # Poisson Binomial
    # -----------------------------
    def poisson_binomial_pmf(self, probs):
        probs = np.array(probs)
        n = len(probs)

        pmf = np.zeros(n + 1)
        pmf[0] = 1.0

        for p in probs:
            pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
            pmf[0] = pmf[0] * (1 - p)

        return pmf

    def poisson_binomial_stats(self, probs):
        probs = np.array(probs)
        mean = probs.sum()
        var = np.sum(probs * (1 - probs))
        return mean, var

    def poisson_binomial_cdf(self, probs):
        pmf = self.poisson_binomial_pmf(probs)
        return np.cumsum(pmf)

    def prob_at_least_k(self, probs, k):
        pmf = self.poisson_binomial_pmf(probs)
        return pmf[k:].sum()

    def prob_at_most_k(self, probs, k):
        pmf = self.poisson_binomial_pmf(probs)
        return pmf[:k + 1].sum()

    def analyze_no_show_risk(self, probs, threshold_k):
        """
        High-level helper for business decisions.
        """
        mean, var = self.poisson_binomial_stats(probs)
        prob_exceed = self.prob_at_least_k(probs, threshold_k)

        return {
            "expected_no_shows": mean,
            "variance": var,
            "prob_at_least_k": prob_exceed
        }

    # -----------------------------
    # Save / Load
    # -----------------------------
    def save_model(self, model_name="model", add_timestamp=True):
        if self.model is None:
            raise ValueError("No model to save.")

        os.makedirs(self.model_folder, exist_ok=True)

        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.joblib"
        else:
            filename = f"{model_name}.joblib"

        path = os.path.join(self.model_folder, filename)
        joblib.dump(self.model, path)

        print(f"Model saved to: {path}")
        return path

    def load_model(self, path):
        self.model = joblib.load(path)
        return self.model

    # -----------------------------
    # Full Pipeline
    # -----------------------------
    def run(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()

        self.train(X_train, y_train)

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = self.evaluate(y_test, y_pred, y_proba)

        return {
            "model": self.model,
            "metrics": metrics,
            "probabilities": y_proba
        }
