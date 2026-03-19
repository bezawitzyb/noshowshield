ORDINAL_FEATURES_MAP = {
    "arrival_date_month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ],
    "meal": [
        "Undefined", "SC", "BB", "HB", "FB"
    ]
}

LEAKY_COLS = ["is_canceled", "reservation_status", "reservation_status_date"]

COLS_TO_DROP ={
    "multicollinearity": ["stays_in_weekend_nights", "stays_in_week_nights", "total_previous_bookings", "total_revenue", "is_last_minute"]
}

COUNTRY_LIMIT = 30

SPLIT_YEAR = 2017

WORKING_MODEL_FILE_NAME = 'working_model.pkl'

WORKING_PREPROCESSOR_FILE_NAME = 'working_preprocessor.pkl'
