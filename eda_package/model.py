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
import pandas as pd
from eda_package.data import load_raw_data, clean_data, temporal_split_v2, temporal_split, split_X_y
from eda_package.features import engineer_features
from eda_package.preprocessor import (
    group_countries,
    get_feature_lists,
    create_preprocessor,
    fit_transform_preprocessor,
    transform_preprocessor
)
from eda_package.registry import ORDINAL_FEATURES_MAP, COUNTRY_LIMIT
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
