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


