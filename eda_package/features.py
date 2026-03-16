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


