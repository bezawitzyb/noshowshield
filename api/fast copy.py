from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from eda_package.model import train_model
from eda_package.preprocessor import transform_preprocessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# PRELIM Load model and preprocessor at startup
model, preprocessor = train_model()

# Preload the model to accelerate the predictions
# app.state.model = load_model()
# Preload the preprocessor to accelerate the predictions
# app.state.preprocessor = load_preprocessor()

# Define request body schema
class PredictionRequest(BaseModel):
    data: dict

# Endpoint for root
@app.get("/")
def root():
    return {
        'message': "Hi, The API of NoshowShield is running! Be ready..."
    }

# Endpoint to predict
@app.get("/predict")
def predict(
    hotel: str,  # Resort Hotel
    lead_time: int,  # 342
    arrival_date_year: int,  # 2015
    arrival_date_month: str,  # July
    arrival_date_week_number: int,  # 27
    arrival_date_day_of_month: int,  # 1
    stays_in_weekend_nights: int,  # 0
    stays_in_week_nights: int,  # 0
    adults: int,  # 2
    children: float,  # 0.0
    babies: int,  # 0
    meal: str,  # BB
    country: str,  # PRT
    market_segment: str,  # Direct
    distribution_channel: str,  # Direct
    is_repeated_guest: int,  # 0
    previous_cancellations: int,  # 0
    previous_bookings_not_canceled: int,  # 0
    reserved_room_type: str,  # C
    assigned_room_type: str,  # C
    booking_changes: int,  # 3
    deposit_type: str,  # No Deposit
    agent: float,  # 0.0
    company: float,  # 0.0
    days_in_waiting_list: int,  # 0
    customer_type: str,  # Transient
    adr: float,  # 0.0
    required_car_parking_spaces: int,  # 0
    total_of_special_requests: int,  # 0
    country_group: str,  # PRT
    total_stay_nights: int,  # 0
    has_children: int,  # 0
    is_last_minute: int,  # 0
    season: str,  # Summer
    total_previous_bookings: int,  # 0
    prev_cancel_ratio: float,  # -1.0
    total_revenue: float,  # 0.0
    segment_cancel_rate: float,  # 0.14715350728566587
    has_deposit: int,  # 0
    arrival_date: str  # 2015-07-01 00:00:00
):
    #Define model and preprocessor at the beginning of the function to ensure they are loaded
    model = model
    #model = app.state.model
    preprocessor = preprocessor
    #preprocessor = app.state.preprocessor
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

# Endpoint for POST https://your-domain.com/predict
# POST: {"data": {feature1: value1, feature2: value2, ..., feature40: value40}}
@app.get("/predict")
def predict_post(request: PredictionRequest):
    """POST endpoint - pass all 40 features as JSON body"""
    return _predict(request.data)

def _predict(data: dict):
    """Shared prediction logic"""
    try:
        # Convert input dictionary to DataFrame
        X = pd.DataFrame([data])

        # Transform using the preprocessor
        X_processed = transform_preprocessor(X, preprocessor)

        # Get prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]

        return {
            'prediction': int(prediction),
            'probability': {
                'no_cancel': float(probability[0]),
                'cancel': float(probability[1])
            },
            'inputs': data
        }
    except Exception as e:
        return {
            'error': str(e),
            'inputs': data
        }
