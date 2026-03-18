import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from eda_package.model import train_model
from eda_package.preprocessor import transform_preprocessor

from eda_package.main import pred

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
    X_pred = pd.DataFrame([{
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests
    }])

    return pred(X_pred)
