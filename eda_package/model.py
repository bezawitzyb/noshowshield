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
