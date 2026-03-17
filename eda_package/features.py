"""
NoShowShield — Feature engineering pipeline.

Responsibilities:
    - Create new features from raw columns
    - Each transform is an independent function (testable, composable)
    - engineer_all_features() runs the full pipeline in correct order

Usage:
    from noshowshield.ml_logic.features import engineer_all_features

    df = engineer_all_features(df)

Design principles:
    - Every function takes a DataFrame and returns a DataFrame (chainable)
    - No function drops columns — that's data.py's job
    - Each function is idempotent: running it twice produces the same result
    - Functions are ordered: rolling features depend on temporal sorting
"""


import pandas as pd
import numpy as np


def add_stay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stay-duration and guest-composition features.

    New columns:
        total_stay_nights  — weekend + weekday nights
        has_children       — flag for children or babies present
    """
    df = df.copy()

    df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['has_children'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)

    return df


def add_lead_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lead-time derived features.

    New columns:
        is_last_minute    — booked within 3 days of arrival
    """
    df = df.copy()
    df['is_last_minute'] = (df['lead_time'] <= 3).astype(int)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and seasonal features from arrival_date.

    Requires 'arrival_date' column (datetime).

    New columns:
        season                — Winter / Spring / Summer / Autumn
    """
    df = df.copy()

    month_to_season = {
        'January': 'Winter', 'February': 'Winter', 'March': 'Spring',
        'April': 'Spring', 'May': 'Spring', 'June': 'Summer',
        'July': 'Summer', 'August': 'Summer', 'September': 'Autumn',
        'October': 'Autumn', 'November': 'Autumn', 'December': 'Winter'
    }
    df['season'] = df['arrival_date_month'].map(month_to_season)

    return df


def add_booking_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer history and engagement features.

    New columns:
        total_previous_bookings — cancelled + not-cancelled history
        prev_cancel_ratio       — fraction of past bookings cancelled (-1 = no history)
    """
    df = df.copy()

    df['total_previous_bookings'] = (
        df['previous_cancellations'] + df['previous_bookings_not_canceled']
    )
    df['prev_cancel_ratio'] = np.where(
        df['total_previous_bookings'] > 0,
        df['previous_cancellations'] / df['total_previous_bookings'],
        -1  # -1 signals "no history" — a distinct category for the model
    )

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price and revenue features from ADR.

    Requires 'total_guests' and 'total_stay_nights' (call add_stay_features first).

    New columns:
        total_revenue   — ADR * max(total_stay_nights, 1)
    """
    df = df.copy()
    df['total_revenue'] = df['adr'] * np.maximum(df['total_stay_nights'], 1)

    return df


def add_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create market-segment features.

    New columns:
        segment_cancel_rate — historical cancel rate for this segment
    """
    df = df.copy()

    df['segment_cancel_rate'] = df.groupby('market_segment')['is_canceled'].transform('mean')

    return df


def add_deposit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deposit and financial commitment features.

    New columns:
        has_deposit — any deposit type other than 'No Deposit'
    """
    df = df.copy()

    df['has_deposit'] = (df['deposit_type'] != 'No Deposit').astype(int)

    return df


# Combined feature engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in the correct order.

    This is the single entry point for all feature creation. It calls
    each feature group function sequentially and returns the enriched
    dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with leakage columns already removed

    Returns
    -------
    pd.DataFrame
        Dataframe with all 9 engineered features added.
    """
    df = add_stay_features(df)
    df = add_lead_time_features(df)
    df = add_calendar_features(df)
    df = add_booking_history_features(df)
    df = add_price_features(df)
    df = add_segment_features(df)
    df = add_deposit_features(df)

    return df
