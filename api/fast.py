# For local testing:
# uvicorn api.fast:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd

from eda_package.data import DataManager
from eda_package.features import FeatureEngineer
#from eda_package.pipeline import run_from_saved_model
from eda_package.preprocessor import PreprocessorManager
from eda_package.model import ModelManager
from eda_package.optimiser import OverbookingOptimizer
from eda_package.explainer import ExplainerManager

# --- Instantiate once (shared across app) ---
data_manager = DataManager()
feature_engineer = FeatureEngineer()
preprocessor_manager = PreprocessorManager()
model_manager = ModelManager()
explainer_manager = ExplainerManager()

# --- Cache for heavy optimisation artifacts ---
optimisation_cache: dict = {}
explainability_cache: dict = {}

def train_artifacts_once():
    """
    Train preprocessor + model once if they do not exist yet.
    """
    X_train, X_test, y_train, y_test = data_manager.prepare_train_test_data()

    X_train = feature_engineer.engineer_features(X_train)
    X_test = feature_engineer.engineer_features(X_test)

    X_train_processed, X_test_processed, _ = preprocessor_manager.prepare_train_test(
        X_train, X_test
    )

    model_manager.train(X_train_processed, y_train)

    preprocessor_manager.save()
    model_manager.save()

def prepare_optimisation_artifacts_once() -> None:
    print("Preparing optimisation artifacts...")

    X_train, X_test, y_train, y_test = data_manager.prepare_train_test_data()
    print("Data prepared")

    X_train_fe = feature_engineer.engineer_features(X_train.copy())
    X_test_fe = feature_engineer.engineer_features(X_test.copy())
    print("Features engineered")

    X_train_processed = preprocessor_manager.transform(X_train_fe)
    X_test_processed = preprocessor_manager.transform(X_test_fe)
    print("Preprocessing done")

    cancel_probs = model_manager.predict_proba(X_test_processed)[:, 1]
    metrics = model_manager.evaluate(X_test_processed, y_test)
    print("Predictions done")

    optimizer = OverbookingOptimizer()

    X_test_with_dates = optimizer.build_arrival_date(X_test.copy())
    print("Test arrival dates built")

    X_train_with_dates = optimizer.build_arrival_date(X_train.copy())
    X_train_with_dates["is_canceled"] = y_train
    print("Train arrival dates built")

    capacity_map = optimizer.infer_capacity(X_train_with_dates)
    print("Capacity inferred")

    X_test_with_dates["is_canceled"] = y_test

    model = model_manager.model
    try:
        model_params = {}
    except AttributeError:
        model_params = {"note": "params unavailable (XGBoost version mismatch)"}

    model_info = {
        "model_type": type(model).__name__,
        "model_params": model_params,
    }

    optimisation_cache["X_test_with_dates"] = X_test_with_dates
    optimisation_cache["cancel_probs"] = cancel_probs
    optimisation_cache["capacity_map"] = capacity_map
    optimisation_cache["metrics"] = metrics
    optimisation_cache["model_info"] = model_info

    print("Optimisation cache ready")
    print(optimisation_cache.keys())

def prepare_explainability_artifacts_once() -> None:
    """
    Prepare SHAP background data once and build the explainer.
    """
    X_train, X_test, y_train, y_test = data_manager.prepare_train_test_data()

    X_train_fe = feature_engineer.engineer_features(X_train.copy())
    X_test_fe = feature_engineer.engineer_features(X_test.copy())

    X_train_processed = preprocessor_manager.transform(X_train_fe)
    X_test_processed = preprocessor_manager.transform(X_test_fe)

    feature_names = preprocessor_manager.preprocessor.get_feature_names_out()

    X_train_shap = explainer_manager.transform_to_shap_df(
        X_processed=X_train_processed,
        feature_names=feature_names,
        index=X_train.index,
    )

    X_test_shap = explainer_manager.transform_to_shap_df(
        X_processed=X_test_processed,
        feature_names=feature_names,
        index=X_test.index,
    )

    background = X_train_shap.iloc[:50]
    explainer_manager.build_explainer(model_manager, background)

    explainability_cache["X_train"] = X_train
    explainability_cache["X_test"] = X_test
    explainability_cache["X_test_shap"] = X_test_shap
    explainability_cache["feature_names"] = feature_names


# --- Lifespan handler ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    try:
        try:
            preprocessor_manager.load()
            model_manager.load()
        except FileNotFoundError:
            train_artifacts_once()
            preprocessor_manager.load()
            model_manager.load()

        prepare_optimisation_artifacts_once()
        prepare_explainability_artifacts_once()

    except Exception as e:
        print(f"Startup error: {repr(e)}")
        raise

    yield
    # SHUTDOWN (optional cleanup if needed)


app = FastAPI(lifespan=lifespan)


# --- Request schema ---
class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float | int | None = 0
    babies: int = 0
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float | int | None = 0
    company: float | int | None = 0
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int


# --- Routes ---
@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(booking: BookingInput):
    X_pred = pd.DataFrame([booking.model_dump()])

    X_pred = data_manager.group_countries(X_pred)
    X_pred = feature_engineer.engineer_features(X_pred)
    X_pred_processed = preprocessor_manager.transform(X_pred)

    y_pred = model_manager.predict(X_pred_processed)
    y_prob = model_manager.predict_proba(X_pred_processed)[:, 1]

    return {
        "prediction": int(y_pred[0]),
        "cancellation_probability": float(y_prob[0]),
    }

import time

@app.get("/optimise")
def optimise(
    relocation_cost: float,
    max_risk: float,
) -> dict:
    start = time.time()

    if "X_test_with_dates" not in optimisation_cache:
        prepare_optimisation_artifacts_once()

    print(f"Cache check: {time.time() - start:.2f}s")

    optimizer = OverbookingOptimizer(
        relocation_cost=relocation_cost,
        max_risk=max_risk,
        max_extra_sweep=100,
    )

    t1 = time.time()

    recommendations = optimizer.aggregate_and_recommend(
        raw_df=optimisation_cache["X_test_with_dates"],
        cancel_probs=optimisation_cache["cancel_probs"],
        capacity_map=optimisation_cache["capacity_map"],
    )

    t2 = time.time()
    print(f"aggregate_and_recommend: {t2 - t1:.2f}s")
    print(f"total optimise endpoint: {t2 - start:.2f}s")

    return {
        "model_info": optimisation_cache["model_info"],
        "metrics": optimisation_cache["metrics"],
        "recommendations": recommendations.to_dict("records"),
    }


# @app.get("/optimise")
# def optimise(
#     relocation_cost: float,
#     max_risk: float
# )-> dict:

#     user_input = {
#         "relocation_cost": relocation_cost,
#         "max_risk": max_risk,
#         "max_extra_sweep": 500,
#         "model_file_name": WORKING_MODEL_FILE_NAME,
#         "preprocessor_file_name": "preprocessor.joblib",
#     }

#     result = run_from_saved_model(**user_input)
#     result["recommendations"] = result["recommendations"].to_dict('records')

#     return result

@app.post("/explain/local")
def explain_local(booking: BookingInput) -> dict:
    X_pred = pd.DataFrame([booking.model_dump()])

    X_pred = data_manager.group_countries(X_pred)
    X_pred = feature_engineer.engineer_features(X_pred)
    X_pred_processed = preprocessor_manager.transform(X_pred)

    feature_names = preprocessor_manager.preprocessor.get_feature_names_out()

    X_pred_shap = explainer_manager.transform_to_shap_df(
        X_processed=X_pred_processed,
        feature_names=feature_names,
        index=X_pred.index,
    )

    local_result = explainer_manager.explain_local(X_pred_shap, row_index=0)
    grouped_local = local_result["grouped_local_shap"]

    higher_risk, lower_risk = explainer_manager.split_local_drivers(
        grouped_local,
        top_n=5,
    )

    y_prob = model_manager.predict_proba(X_pred_processed)[:, 1]

    return {
        "cancellation_probability": float(y_prob[0]),
        "higher_cancellation_risk": higher_risk.to_dict("records"),
        "lower_cancellation_risk": lower_risk.to_dict("records"),
        "grouped_local_shap": grouped_local.to_dict("records"),
    }

@app.get("/explain/global-by-date")
def explain_global_by_date(
    selected_date: str,
    room_type: str | None = None,
    min_rows: int = 5,
) -> dict:
    X_raw = explainability_cache["X_test"]

    # Filter by room type if provided
    if room_type is not None:
        X_raw = X_raw[X_raw["assigned_room_type"] == room_type]

    result = explainer_manager.explain_global_for_date(
        selected_date=selected_date,
        X_raw=X_raw,
        data_manager=data_manager,
        feature_engineer=feature_engineer,
        preprocessor_manager=preprocessor_manager,
        min_rows=min_rows,
    )

    response = {
        "selected_date": str(result["selected_date"].date()),
        "n_bookings": result["n_bookings"],
        "message": result["message"],
        "grouped_global_shap": None,
    }

    if result["grouped_global_shap"] is not None:
        response["grouped_global_shap"] = result["grouped_global_shap"].to_dict("records")

    return response

@app.get("/explain/available-dates")
def explain_available_dates() -> dict:
    X_test = explainability_cache["X_test"].copy()

    X_test["arrival_date"] = pd.to_datetime(
        X_test["arrival_date_day_of_month"].astype(str)
        + " "
        + X_test["arrival_date_month"].astype(str)
        + " "
        + X_test["arrival_date_year"].astype(str),
        format="%d %B %Y",
        errors="coerce"
    )

    counts = (
        X_test["arrival_date"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["arrival_date", "n_bookings"]
    counts["arrival_date"] = counts["arrival_date"].dt.strftime("%Y-%m-%d")

    return {
        "dates": counts.to_dict("records")
    }
