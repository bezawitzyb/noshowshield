from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd

from eda_package.data import DataManager
from eda_package.features import FeatureEngineer
from eda_package.preprocessor import PreprocessorManager
from eda_package.model import ModelManager


# --- Instantiate once (shared across app) ---
data_manager = DataManager()
feature_engineer = FeatureEngineer()
preprocessor_manager = PreprocessorManager()
model_manager = ModelManager()


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


# --- Lifespan handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    try:
        preprocessor_manager.load()
        model_manager.load()
    except FileNotFoundError:
        train_artifacts_once()

    yield  # app runs here

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
