import streamlit as st
import pandas as pd

from eda_package import *



st.set_page_config(page_title="Overbooking Optimizer", layout="wide")

st.title("Hotel Overbooking Optimizer")


st.sidebar.header("Optimization Settings")

relocation_cost = st.sidebar.number_input(
    "Relocation cost (€)",
    min_value=0.0,
    max_value=1000.0,
    value=300.0,   # default
    step=50.0
)

max_risk = st.sidebar.slider(
    "Max relocation risk",
    min_value=0.0,
    max_value=0.1,
    value=0.02,   # default = 5%
    step=0.01
)

# -------------------------------------------------
# LOAD PIPELINE + DATA (cached so it runs once)
# -------------------------------------------------
@st.cache_resource
def load_pipeline_and_results(relocation_cost, max_risk):
    pipe = SimpleModelPipeline(
        relocation_cost=relocation_cost,
        max_risk=max_risk
    )

    results = pipe.run_from_saved_model("/Users/beza/code/bezawitzyb/noshowshield/models/simple_logistic_model_20260318_131522.joblib")

    return pipe, results

pipe, results = load_pipeline_and_results(relocation_cost, max_risk)
recs = results["recommendations"]

st.caption(
    f"Using relocation cost = €{relocation_cost:.0f}, "
    f"max relocation risk = {max_risk*100:.1f}%"
)

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

# Date selector
available_dates = sorted(recs["arrival_date"].dt.date.unique())
selected_date = st.sidebar.selectbox("Select date", available_dates)

# Room type selector
available_rooms = sorted(recs["assigned_room_type"].unique())
selected_room = st.sidebar.selectbox("Select room type", available_rooms)

# -------------------------------------------------
# FILTER DATA
# -------------------------------------------------
filtered = pipe.get_recommendations(
    recs,
    dates=[selected_date],
    room_types=[selected_room]
)

# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------
st.subheader("Recommendation")

if filtered.empty:
    st.warning("No data available for this selection.")
else:
    row = filtered.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Capacity", int(row["capacity"]))
    col2.metric("Current Bookings", int(row["total_bookings"]))
    col3.metric("Expected Show-ups", round(row["expected_show_ups"], 1))

    st.divider()

    col4, col5, col6 = st.columns(3)

    col4.metric("Recommended Extra Bookings", int(row["recommended_extra"]))
    col5.metric("Net Benefit (€)", row["net_benefit"])
    col6.metric("Relocation Risk", f"{row['relocation_probability']*100:.2f}%")

    st.divider()

    st.subheader("Detailed View")

    # Columns you want to show
    display_cols = [
        "arrival_date",
        "assigned_room_type",
        "capacity",
        "total_bookings",
        "expected_show_ups",
        "expected_cancellations",
        "recommended_extra",
        "net_benefit",
        "relocation_probability",
    ]

    # Create display dataframe
    nice_df = filtered[display_cols].rename(columns={
        "arrival_date": "Date",
        "assigned_room_type": "Room",
        "capacity": "Capacity",
        "total_bookings": "Bookings",
        "expected_show_ups": "Expected Show-ups",
        "expected_cancellations": "Expected Cancels",
        "recommended_extra": "Recommended additional booking",
        "net_benefit": "Net €",
        "relocation_probability": "Relocation Risk",
    })

    # Show nicely formatted table
    st.dataframe(
        nice_df.style.format({
            "Expected Show-ups": "{:.1f}",
            "Expected Cancels": "{:.1f}",
            "Net €": "€{:.2f}",
            "Relocation Risk": "{:.2%}",
        }),
        use_container_width=True
    )
