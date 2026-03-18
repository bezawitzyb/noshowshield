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
import pickle
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

class BookingPredictor():

    def __init__(self,
                 file_name: str = None,
                 X_train_processed: pd.DataFrame = None,
                 y_train: pd.Series = None):

        if X_train_processed is not None and y_train is not None:
            self.train_model(X_train_processed, y_train)
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
        url = '../models/' + file_name
        #print('Loading: ', url)
        self.model = pickle.load(open(url, 'rb'))



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
