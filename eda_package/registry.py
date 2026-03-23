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

FRONT_END_PERIODS = [('2015-01-02','2015-01-01'), ('2017-07-01','2017-08-31')]

DEFAULT_RELOCATION_COST = 300
DEFAULT_MAX_RISK= 0.02

WORKING_MODEL_FILE_NAME = 'working_model.pkl'
MAX_EXTRA_SWEEP=1000

RELEVANT_FEATURES = ['num__lead_time',
    'num__adr',
    'num__arrival_date_day_of_month',
    'num__arrival_date_week_number',
    'num__agent',
    'num__stays_in_week_nights',
    'cat_ordinal__arrival_date_month',
    'num__weekend_ratio',
    'num__arrival_date_year',
    'num__total_of_special_requests',
    'num__stays_in_weekend_nights',
    'num__special_requests_per_guest',
    'num__booking_changes',
    'cat_ordinal__meal',
    'num__adults',
    'num__company',
    'cat_onehot__hotel_City Hotel',
    'cat_onehot__reserved_room_type_A',
    'cat_onehot__assigned_room_type_A',
    'cat_onehot__country_PRT',
    'cat_onehot__assigned_room_type_D',
    'num__children',
    'num__previous_bookings_not_canceled',
    'cat_onehot__customer_type_Transient',
    'cat_onehot__market_segment_Online TA',
    'cat_onehot__customer_type_Transient-Party',
    'cat_onehot__reserved_room_type_D',
    'cat_onehot__distribution_channel_TA/TO',
    'cat_onehot__market_segment_Direct',
    'cat_onehot__distribution_channel_Direct']
