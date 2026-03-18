import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from .data import load_raw_data, clean_data, temporal_split_v2, split_X_y
from .features import engineer_features_v2
from .preprocessor import (
    get_feature_lists,
    create_preprocessor,
    fit_transform_preprocessor,
    transform_preprocessor,
    group_countries
)
from .registry import ORDINAL_FEATURES_MAP, COUNTRY_LIMIT
from .features import add_segment_cancel_rate

def train():
    """
    Train model on training set and return fitted model + fitted preprocessor.
    """
    df = load_raw_data()
    df = clean_data(df)
    df = group_countries(df, COUNTRY_LIMIT)
    df = engineer_features_v2(df)

    training_set, test_set = temporal_split_v2(df, 2017, 3)

    X_train, y_train = split_X_y(training_set)
    X_test, y_test = split_X_y(test_set)

    feature_lists = get_feature_lists(X_train, ORDINAL_FEATURES_MAP)
    preprocessor = create_preprocessor(feature_lists, ORDINAL_FEATURES_MAP)

    X_train_processed = fit_transform_preprocessor(X_train, preprocessor)
    X_test_processed = transform_preprocessor(X_test, preprocessor)

    model = LogisticRegression(max_iter=3000, random_state=42)
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 2),
        "recall": round(recall_score(y_test, y_pred), 2),
        "precision": round(precision_score(y_test, y_pred), 2),
        "f1": round(f1_score(y_test, y_pred), 2),
        "auc": round(roc_auc_score(y_test, y_prob), 2),
    }

    print(metrics)

    return model, preprocessor, metrics


def pred(X_pred: pd.DataFrame):
    """
    Make prediction on new booking data using trained model + fitted preprocessor.
    """
    model, preprocessor, _ = train()

    X_pred = group_countries(X_pred, COUNTRY_LIMIT)
    X_pred = engineer_features_v2(X_pred)
    X_pred_processed = transform_preprocessor(X_pred, preprocessor)

    y_pred = model.predict(X_pred_processed)
    y_prob = model.predict_proba(X_pred_processed)[:, 1]

    return {
        "prediction": int(y_pred[0]),
        "cancellation_probability": float(y_prob[0])
    }
