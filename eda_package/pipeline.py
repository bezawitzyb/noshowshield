"""
NoShowShield — End-to-end inference pipeline

Responsibilities:
    - Wire together all package components
    - Load saved model + preprocessor
    - Run predictions and generate overbooking recommendations

Usage:
    from eda_package.pipeline import run_from_saved_model

    results = run_from_saved_model(
        relocation_cost=300,
        max_risk=0.02,
    )
    recommendations = results["recommendations"]
    metrics = results["metrics"]
"""

from typing import Optional
from eda_package.registry import MAX_EXTRA_SWEEP


def run_from_saved_model(
    relocation_cost: float = 300.0,
    max_risk: float = 0.02,
    max_extra_sweep: int = MAX_EXTRA_SWEEP,
    model_file_name: Optional[str] = None,
    preprocessor_file_name: str = "preprocessor.joblib",
) -> dict:
    """
    Execute the full pipeline using saved model + preprocessor.

    Steps
    -----
    1. Load & prepare data          (DataManager)
    2. Split into train / test       (DataManager)
    3. Engineer features             (FeatureEngineer)
    4. Load preprocessor & transform (PreprocessorManager)
    5. Load model & predict          (ModelManager)
    6. Evaluate metrics
    7. Build arrival dates & infer capacity from training data
    8. Aggregate & recommend on the test set

    Returns
    -------
    dict with keys: model, metrics, recommendations
    """
    from .data import DataManager
    from .features import FeatureEngineer
    from .preprocessor import PreprocessorManager
    from .model import ModelManager
    from .optimiser import OverbookingOptimizer

    # 1-2  data
    data_manager = DataManager()
    X_train, X_test, y_train, y_test = data_manager.prepare_train_test_data()

    # 3  feature engineering
    feature_engineer = FeatureEngineer()
    X_train_fe = feature_engineer.engineer_features(X_train)
    X_test_fe = feature_engineer.engineer_features(X_test)

    # 4  preprocessing (load fitted preprocessor)
    preprocessor_manager = PreprocessorManager(file_name=preprocessor_file_name)
    preprocessor_manager.load()
    X_train_processed = preprocessor_manager.transform(X_train_fe)
    X_test_processed = preprocessor_manager.transform(X_test_fe)

    # 5  model (load trained model)
    model_manager = ModelManager(
        file_name=model_file_name
    ) if model_file_name else ModelManager()
    model_manager.load()

    # 6  evaluate
    metrics = model_manager.evaluate(X_test_processed, y_test)

    # 7  cancellation probabilities on test set
    cancel_probs = model_manager.predict_proba(X_test_processed)[:, 1]

    # 8  optimizer + arrival dates
    optimizer = OverbookingOptimizer(
        relocation_cost=relocation_cost,
        max_risk=max_risk,
        max_extra_sweep=max_extra_sweep,
    )

    X_test_with_dates = optimizer.build_arrival_date(X_test)

    # 9  infer capacity from training data
    X_train_with_dates = optimizer.build_arrival_date(X_train)
    X_train_with_dates["is_canceled"] = y_train
    capacity_map = optimizer.infer_capacity(X_train_with_dates, hotel_col="hotel")

    # 10 aggregate & recommend
    X_test_with_dates["is_canceled"] = y_test
    recommendations = optimizer.aggregate_and_recommend(
        raw_df=X_test_with_dates,
        cancel_probs=cancel_probs,
        capacity_map=capacity_map,
    )

    # 11 collect model metadata for display
    model = model_manager.model
    model_info = {
        "model_type": type(model).__name__,
        "model_params": model.get_params(),
    }

    return {
        "model_info": model_info,
        "metrics": metrics,
        "recommendations": recommendations,
    }


def compute_group_distribution(
    recommendations: "pd.DataFrame",
    raw_df: "pd.DataFrame",
    cancel_probs,
    capacity_map: dict,
    hotel: str,
    arrival_date: str,
    room_type: str,
    n_total: int,
) -> dict:
    """
    Compute the show-up PMF for one (hotel, date, room-type) group.

    Parameters
    ----------
    recommendations : DataFrame returned by run_from_saved_model
    raw_df          : X_test_with_dates (needs 'arrival_date' column)
    cancel_probs    : array of booking-level cancel probabilities
    capacity_map    : {(hotel, room_type): capacity}
    hotel, arrival_date, room_type : group identifiers
    n_total         : total bookings to simulate (slider value)

    Returns
    -------
    dict  –  output of OverbookingOptimizer.compute_showup_distribution
    """
    import numpy as np
    import pandas as pd
    from .optimiser import OverbookingOptimizer

    df = raw_df.copy()
    df["cancel_prob"] = np.asarray(cancel_probs)

    target_date = pd.to_datetime(arrival_date)

    filtered = df[
        (df["hotel"] == hotel)
        & (df["arrival_date"] == target_date)
        & (df["assigned_room_type"] == room_type)
    ]

    if filtered.empty:
        return {"error": "No bookings found for this selection."}

    group_probs = filtered["cancel_prob"].values
    capacity = capacity_map.get((hotel, room_type), len(group_probs))

    optimizer = OverbookingOptimizer()
    return optimizer.compute_showup_distribution(
        cancel_probs=group_probs,
        n_total=n_total,
        capacity=int(capacity),
    )
