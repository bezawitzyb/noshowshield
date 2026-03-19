import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from eda_package.model import train_model
from eda_package.preprocessor import transform_preprocessor

from eda_package.main import pred

#preprocessor_manager.load()
#X_pred_processed = preprocessor_manager.transform(X_pred)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "NoShowShield API is running!"
    }

@app.get("/predict")
def predict(
    hotel: str,
    lead_time: int,
    arrival_date_year: int,
    arrival_date_month: str,
    arrival_date_week_number: int,
    arrival_date_day_of_month: int,
    stays_in_weekend_nights: int,
    stays_in_week_nights: int,
    adults: int,
    children: float,
    babies: int,
    meal: str,
    country: str,
    market_segment: str,
    distribution_channel: str,
    is_repeated_guest: int,
    previous_cancellations: int,
    previous_bookings_not_canceled: int,
    reserved_room_type: str,
    assigned_room_type: str,
    booking_changes: int,
    deposit_type: str,
    agent: float,
    company: float,
    days_in_waiting_list: int,
    customer_type: str,
    adr: float,
    required_car_parking_spaces: int,
    total_of_special_requests: int
):
    # X_pred = pd.DataFrame([{
    #     "hotel": hotel,
    #     "lead_time": lead_time,
    #     "arrival_date_year": arrival_date_year,
    #     "arrival_date_month": arrival_date_month,
    #     "arrival_date_week_number": arrival_date_week_number,
    #     "arrival_date_day_of_month": arrival_date_day_of_month,
    #     "stays_in_weekend_nights": stays_in_weekend_nights,
    #     "stays_in_week_nights": stays_in_week_nights,
    #     "adults": adults,
    #     "children": children,
    #     "babies": babies,
    #     "meal": meal,
    #     "country": country,
    #     "market_segment": market_segment,
    #     "distribution_channel": distribution_channel,
    #     "is_repeated_guest": is_repeated_guest,
    #     "previous_cancellations": previous_cancellations,
    #     "previous_bookings_not_canceled": previous_bookings_not_canceled,
    #     "reserved_room_type": reserved_room_type,
    #     "assigned_room_type": assigned_room_type,
    #     "booking_changes": booking_changes,
    #     "deposit_type": deposit_type,
    #     "agent": agent,
    #     "company": company,
    #     "days_in_waiting_list": days_in_waiting_list,
    #     "customer_type": customer_type,
    #     "adr": adr,
    #     "required_car_parking_spaces": required_car_parking_spaces,
    #     "total_of_special_requests": total_of_special_requests
    # }])

    X_pred = pd.DataFrame([{
        "hotel": "City Hotel",
        "lead_time": 112,
        "arrival_date_year": 2016,
        "arrival_date_month": "December",
        "arrival_date_week_number": 53,
        "arrival_date_day_of_month": 27,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 3,
        "adults": 3,
        "children": 0.0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "D",
        "assigned_room_type": "D",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": 83.0,
        "company": np.nan,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 131.13,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }])

    return pred(X_pred)
