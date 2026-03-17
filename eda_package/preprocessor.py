"""
NoShowShield — Sklearn preprocessing pipeline.

Responsibilities:
    - Build a ColumnTransformer that scales numericals and encodes categoricals
    - Consistent transformation between training and testing data
    - Prevent data leakage: fit only on training data, transform on both

Usage:
    from noshowshield.eda_package.preprocessor import build_preprocessor

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

def group_countries(
    df:pd.DataFrame,
    limit:int
    )-> pd.DataFrame:
    """
    This function takes in a dataframe and group countries with less then 'limit' entries in Other category
    The function adds a new column called 'country_group' and leaves the 'country' column as-is
    """
    df = df.copy()

    country_counts = df['country'].value_counts()
    countries_included = country_counts[country_counts >= limit].index

    df['country_group'] = df['country'].apply(
        lambda x: x if x in countries_included else 'Other'
    )

    #df = df.drop(columns='country')
    #df = df.drop(columns=['country'])

    return df


def get_feature_lists(X: pd.DataFrame, ordinal_feature_map: dict):
    X = X.copy()

    # --- categorical ---
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # --- numeric ---
    numeric_candidates = X.select_dtypes(include=["number"]).columns.tolist()

    binary_features = []
    numerical_features = []

    for col in numeric_candidates:
        unique_vals = set(X[col].dropna().unique())

        if unique_vals.issubset({0, 1}) and len(unique_vals) <= 2:
            binary_features.append(col)
        else:
            numerical_features.append(col)

    # --- split categorical ---
    ordinal_features = [
        col for col in categorical_features if col in ordinal_feature_map
    ]

    onehot_features = [
        col for col in categorical_features if col not in ordinal_feature_map
    ]

    return {
        "numerical_features": numerical_features,
        "binary_features": binary_features,
        "onehot_features": onehot_features,
        "ordinal_features": ordinal_features
    }

def create_preprocessor(feature_lists: dict, ordinal_features_map: dict) -> ColumnTransformer:
    """
    Create an unfitted ColumnTransformer preprocessor.

    Parameters
    ----------
    feature_lists : dict
        Output of get_feature_lists(), containing:
        - numerical_features
        - binary_features
        - onehot_features
        - ordinal_features

    ordinal_features_map : dict
        Mapping of ordinal feature name -> ordered categories
    """
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
                    categories=[ordinal_features_map[col] for col in ordinal_features]
                )
            )
        ])
        transformers.append(("cat_ordinal", ordinal_pipeline, ordinal_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def fit_transform_preprocessor(X_train: pd.DataFrame, preprocessor):
    X_train_processed = preprocessor.fit_transform(X_train)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    return X_train_processed

def transform_preprocessor(X_test: pd.DataFrame, preprocessor):
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    return X_test_processed
    return preproc_pipeline

#df = engineer_numerical_features(df)
#numeric_features, binary_features = get_feature_lists(df)
#preproc_pipeline = build_preprocessing_pipeline(
   # numeric_features=numeric_features,
   # binary_features=binary_features)


#define a function that takes the dataframe df and outputs a fully preprocessed dataframe ready for modeling. This function should call the previous functions in the correct order.
def preprocess_data(df: pd.DataFrame, ordinal_feature_map: dict, country_limit: int) -> pd.DataFrame:
    df = df.copy()

    #Group countries with less than 'country_limit' entries into 'Other'
    df = group_countries(df, country_limit)

    #Get feature lists
    feature_lists = get_feature_lists(df, ordinal_feature_map)

    #Create preprocessor
    preprocessor = create_preprocessor(feature_lists, ordinal_feature_map)

    #Fit and transform the preprocessor on the entire dataframe (since this function is meant for preprocessing before modeling, we assume this is the training data)
    df_processed = fit_transform_preprocessor(df, preprocessor)

    return df_processed



def preproc_test():
    print("Hello from preprocessor!")
