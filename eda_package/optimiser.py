"""
NoShowShield — Overbooking optimiser (Poisson-Binomial)

Responsibilities:
    - Compute the full cancellation/show-up probability distribution
    - Infer room-type capacity from historical booking counts
    - Aggregate booking-level cancellation probabilities
    - Find the optimal number of extra bookings to accept
    - Compute financial impact (revenue vs relocation cost)
    - Run full pipeline from a saved model

Usage:
    from eda_package.optimiser import OverbookingOptimizer

    optimizer = OverbookingOptimizer()

    # --- Option A: manual step-by-step ---
    df_with_dates = optimizer.build_arrival_date(raw_df)
    capacity_map  = optimizer.infer_capacity(df_with_dates)
    recommendations = optimizer.aggregate_and_recommend(
        raw_df=df_with_dates,
        cancel_probs=cancel_probs,
        capacity_map=capacity_map,
    )

    # --- Option B: end-to-end from saved artefacts ---
    results = optimizer.run_from_saved_model()
"""

from typing import Optional, Dict, Sequence

import numpy as np
import pandas as pd

from .registry import DEFAULT_RELOCATION_COST, DEFAULT_MAX_RISK, MAX_EXTRA_SWEEP


class OverbookingOptimizer:
    """
    Optimizer for overbooking recommendations using booking-level
    cancellation probabilities and a Poisson-Binomial distribution.
    """

    def __init__(
        self,
        relocation_cost: float = DEFAULT_RELOCATION_COST,
        max_risk: float = DEFAULT_MAX_RISK,
        max_extra_sweep: int = MAX_EXTRA_SWEEP,
    ):
        self.relocation_cost = relocation_cost
        self.max_risk = max_risk
        self.max_extra_sweep = max_extra_sweep

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def build_arrival_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compose a datetime column from arrival year, month name, and day.
        """
        df = df.copy()
        df["arrival_date"] = pd.to_datetime(
            df["arrival_date_year"].astype(str)
            + "-"
            + df["arrival_date_month"]
            + "-"
            + df["arrival_date_day_of_month"].astype(str),
            format="%Y-%B-%d",
        )
        return df

    def infer_capacity(
        self,
        df: pd.DataFrame,
        hotel_col: str = "hotel",
        date_col: str = "arrival_date",
        room_col: str = "assigned_room_type",
    ) -> Dict:
        """
        Infer room capacity from historical data.

        Count only bookings that actually showed up (is_canceled == 0).
        The maximum observed show-ups on any single date is used as a
        tight lower bound on true physical capacity.
        """
        showed_up = df[df["is_canceled"] == 0].copy()

        counts = (
            showed_up
            .groupby([date_col, hotel_col, room_col])
            .size()
            .reset_index(name="n_showups")
        )

        capacity_map = (
            counts.groupby([hotel_col,room_col])["n_showups"]
            .max()
            .to_dict()
        )

        return capacity_map

    # ------------------------------------------------------------------
    # Poisson-Binomial helpers
    # ------------------------------------------------------------------

    @staticmethod
    def poisson_binomial_pmf(probs: Sequence[float]) -> np.ndarray:
        """
        Exact Poisson-Binomial PMF via dynamic programming.
        For small n (< 80). O(n²) complexity.
        """
        probs = np.asarray(probs, dtype=np.float64)
        n = len(probs)

        pmf = np.zeros(n + 1)
        pmf[0] = 1.0

        for p in probs:
            new = np.empty_like(pmf)
            new[0] = pmf[0] * (1 - p)
            new[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
            pmf = new

        return pmf

    @staticmethod
    def poisson_binomial_pmf_fft(probs: Sequence[float]) -> np.ndarray:
        """
        Compute Poisson-Binomial PMF using FFT (O(n log n)).
        Much faster for large n (≥80).
        """
        probs = np.asarray(probs, dtype=np.float64)
        n = len(probs)

        # Use characteristic function and FFT
        omega = np.exp(2j * np.pi / (n + 1))
        k = np.arange(n + 1)

        # Characteristic function: prod((1-p) + p*omega^k)
        cf = np.prod(1 - probs[:, None] + probs[:, None] * omega ** k, axis=0)
        pmf = np.fft.ifft(cf).real

        # Ensure non-negative and normalized (numerical safety)
        pmf = np.maximum(pmf, 0)
        pmf = pmf / pmf.sum()

        return pmf

    @staticmethod
    def poisson_binomial_stats(probs: Sequence[float]):
        """
        Mean and variance for total Bernoulli successes.
        """
        probs = np.asarray(probs, dtype=np.float64)
        mean = probs.sum()
        var = (probs * (1 - probs)).sum()
        return mean, var

    # ------------------------------------------------------------------
    # single-group optimisation
    # ------------------------------------------------------------------

    def optimise_group(
        self,
        cancel_probs: Sequence[float],
        capacity: int,
        mean_adr: float,
    ) -> dict:
        """
        Find the best extra-booking level for one (date, room-type) group.
        """
        cancel_probs = np.asarray(cancel_probs, dtype=np.float64)
        n_current = len(cancel_probs)
        show_probs = 1.0 - cancel_probs

        best = {
            "recommended_extra": 0,
            "recommended_total": n_current,
            "net_benefit": 0.0,
            "additional_revenue": 0.0,
            "expected_relocation_cost": 0.0,
            "relocation_probability": 0.0,
        }

        for extra in range(0, self.max_extra_sweep + 1):
            total = n_current + extra

            if extra > 0:
                mean_cancel = cancel_probs.mean()
                extended_show_probs = np.concatenate([
                    show_probs,
                    np.full(extra, 1.0 - mean_cancel),
                ])
            else:
                extended_show_probs = show_probs

            show_pmf = self.poisson_binomial_pmf(extended_show_probs)

            if capacity + 1 <= total:
                relocation_probability = show_pmf[capacity + 1:].sum()
            else:
                relocation_probability = 0.0

            if relocation_probability > self.max_risk:
                break

            expected_excess = sum(
                (k - capacity) * show_pmf[k]
                for k in range(capacity + 1, total + 1)
            )

            additional_revenue = extra * mean_adr
            expected_relocation_cost = expected_excess * self.relocation_cost
            net_benefit = additional_revenue - expected_relocation_cost

            if net_benefit >= best["net_benefit"]:
                best = {
                    "recommended_extra": extra,
                    "recommended_total": total,
                    "net_benefit": round(net_benefit, 2),
                    "additional_revenue": round(additional_revenue, 2),
                    "expected_relocation_cost": round(expected_relocation_cost, 2),
                    "relocation_probability": round(relocation_probability, 4),
                }

        return best

    # ------------------------------------------------------------------
    # aggregate across all groups
    # ------------------------------------------------------------------

    def aggregate_and_recommend(
        self,
        raw_df: pd.DataFrame,
        cancel_probs: Sequence[float],
        capacity_map: Dict,
        group_cols: tuple = ("arrival_date", "hotel", "assigned_room_type"),
        room_col: str = "assigned_room_type",
        hotel_col: str = "hotel",
        adr_col: str = "adr",
    ) -> pd.DataFrame:
        """
        Aggregate booking-level probabilities and compute group-level
        overbooking recommendations.
        """
        df = raw_df.copy()
        df["cancel_prob"] = np.asarray(cancel_probs)

        grouped = (
            df.groupby(list(group_cols))
            .agg(
                total_bookings=("cancel_prob", "size"),
                expected_cancellations=("cancel_prob", "sum"),
                cancel_prob_mean=("cancel_prob", "mean"),
                cancel_prob_std=("cancel_prob", "std"),
                mean_adr=(adr_col, "mean"),
                individual_probs=("cancel_prob", list),
            )
            .reset_index()
        )

        grouped["variance"] = grouped["individual_probs"].apply(
            lambda ps: np.sum(np.array(ps) * (1 - np.array(ps)))
        )
        grouped["std_cancellations"] = np.sqrt(grouped["variance"])

        grouped["ci_lower"] = (
            grouped["expected_cancellations"] - 1.96 * grouped["std_cancellations"]
        ).clip(lower=0)

        grouped["ci_upper"] = (
            grouped["expected_cancellations"] + 1.96 * grouped["std_cancellations"]
        )

        grouped["expected_show_ups"] = (
            grouped["total_bookings"] - grouped["expected_cancellations"]
        )

        grouped["capacity"] = grouped.apply(
            lambda row: capacity_map.get(
                (row[hotel_col], row[room_col]),
                row["total_bookings"]   # fallback if unseen combination
            ),
            axis=1,
        ).astype(int)

        grouped["expected_cancellations"] = grouped["expected_cancellations"].round().astype(int)
        grouped["expected_show_ups"] = grouped["expected_show_ups"].round().astype(int)

        recommendations = []
        for _, row in grouped.iterrows():
            rec = self.optimise_group(
                cancel_probs=row["individual_probs"],
                capacity=row["capacity"],
                mean_adr=row["mean_adr"],
            )
            recommendations.append(rec)

        rec_df = pd.DataFrame(recommendations)

        result = pd.concat(
            [grouped.reset_index(drop=True), rec_df],
            axis=1,
        )

        leading = list(group_cols) + [
            "capacity",
            "total_bookings",
            "expected_cancellations",
            "std_cancellations",
            "ci_lower",
            "ci_upper",
            "expected_show_ups",
            "mean_adr",
            "individual_probs",
            "recommended_extra",
            "recommended_total",
            "additional_revenue",
            "expected_relocation_cost",
            "net_benefit",
            "relocation_probability",
        ]

        extra_cols = [c for c in result.columns if c not in leading]
        result = result[[c for c in leading if c in result.columns] + extra_cols]

        return result

    # ------------------------------------------------------------------
    # cancellation distribution helper
    # ------------------------------------------------------------------
    def get_cancellation_distributions(self, row: pd.Series) -> dict:
        """
        Compute Poisson-Binomial PMF for cancellation counts.
        Uses FFT for large hotels (>80 bookings) for speed.
        """
        cancel_probs = np.array(row["individual_probs"])
        mean_cancel = cancel_probs.mean()
        recommended_extra = int(row["recommended_extra"])
        recommended_total = int(row["recommended_total"])
        capacity = int(row["capacity"])

        n_current = len(cancel_probs)

        # Choose method based on size (FFT is faster for n > 80)
        pmf_method = self.poisson_binomial_pmf_fft if n_current > 80 else self.poisson_binomial_pmf

        # Current state PMF
        pmf_current = pmf_method(cancel_probs)
        x_current = list(range(len(pmf_current)))

        # Recommended state: append extra bookings at mean cancel rate
        if recommended_extra > 0:
            extended_cancel_probs = np.concatenate([
                cancel_probs,
                np.full(recommended_extra, mean_cancel)
            ])
        else:
            extended_cancel_probs = cancel_probs

        pmf_recommended = pmf_method(extended_cancel_probs)
        x_recommended = list(range(len(pmf_recommended)))

        # Minimum cancellations needed so show-ups don't exceed capacity
        min_cancellations_needed = max(0, recommended_total - capacity)

        return {
            "current": {
                "x": x_current,
                "pmf": pmf_current.tolist(),
                "n_bookings": n_current,
            },
            "recommended": {
                "x": x_recommended,
                "pmf": pmf_recommended.tolist(),
                "n_bookings": recommended_total,
            },
            "min_cancellations_needed": min_cancellations_needed,
            "capacity": capacity,
        }

    # ------------------------------------------------------------------
    # filtering helper
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        recommendations: pd.DataFrame,
        dates,
        room_types,
        hotels=None,
        date_col: str = "arrival_date",
        room_col: str = "assigned_room_type",
        hotel_col: str = "hotel",
    ) -> pd.DataFrame:
        """
        Filter recommendations by arrival date(s) and room type(s).
        """
        if isinstance(dates, str):
            dates = [dates]
        if isinstance(room_types, str):
            room_types = [room_types]

        timestamps = [pd.Timestamp(d) for d in dates]

        mask = (
    recommendations[date_col].isin(timestamps)
    & recommendations[room_col].isin(room_types)
)

        if hotels is not None:
            if isinstance(hotels, str):
                hotels = [hotels]
            mask &= recommendations[hotel_col].isin(hotels)

        filtered = recommendations[mask].reset_index(drop=True)
        return filtered
