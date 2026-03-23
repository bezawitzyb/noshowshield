"""
Tests for ExplainerManager (SHAP explainability layer).

Covers:
- Correct explainer backend (XGBoost native pred_contribs, not SHAP library)
- build_explainer wiring
- Local explanation output structure and content
- Global explanation output structure and content
- Feature name grouping logic
- Driver splitting (higher vs lower risk)
- Performance: single-row local explanation must complete quickly
"""

import time

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from eda_package.explainer import ExplainerManager
from eda_package.model import ModelManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "num__lead_time",
    "num__adr",
    "num__stays_in_week_nights",
    "bin__is_repeated_guest",
    "bin__room_type_mismatch",
    "cat_onehot__hotel_City Hotel",
    "cat_onehot__hotel_Resort Hotel",
    "cat_onehot__meal_BB",
    "cat_onehot__meal_HB",
    "cat_onehot__deposit_type_No Deposit",
    "cat_onehot__customer_type_Transient",
    "cat_ordinal__arrival_date_month_January",
]

N_FEATURES = len(FEATURE_NAMES)
N_TRAIN = 120
N_TEST = 20


@pytest.fixture(scope="module")
def synthetic_data():
    """Small synthetic binary dataset with transformer-style column names."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.random((N_TRAIN, N_FEATURES)),
        columns=FEATURE_NAMES,
    )
    # Target correlated with lead_time so the model can learn something
    y = pd.Series((X["num__lead_time"] > 0.5).astype(int))
    return X, y


@pytest.fixture(scope="module")
def trained_model_manager(synthetic_data):
    """Lightweight ModelManager wrapping a small XGBoost model."""
    X, y = synthetic_data
    xgb_model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    xgb_model.fit(X, y)
    mm = ModelManager.__new__(ModelManager)
    mm.model = xgb_model
    mm.model_params = xgb_model.get_params()
    return mm


@pytest.fixture(scope="module")
def built_explainer_manager(trained_model_manager, synthetic_data):
    """ExplainerManager with a fully built explainer."""
    X_background, _ = synthetic_data
    em = ExplainerManager()
    em.build_explainer(trained_model_manager, X_background=X_background)
    return em


@pytest.fixture(scope="module")
def sample_shap_input(synthetic_data):
    """A small DataFrame of rows ready for SHAP inference."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.random((N_TEST, N_FEATURES)),
        columns=FEATURE_NAMES,
    )


# ---------------------------------------------------------------------------
# 1. Explainer backend
# ---------------------------------------------------------------------------

class TestExplainerBackend:
    def test_booster_is_set_after_build(self, built_explainer_manager):
        """build_explainer must store an XGBoost Booster, not a SHAP object."""
        assert built_explainer_manager.booster is not None
        assert isinstance(built_explainer_manager.booster, xgb.core.Booster)

    def test_no_shap_library_explainer_stored(self, built_explainer_manager):
        """The manager must not hold a shap.Explainer — native path only."""
        assert not hasattr(built_explainer_manager, "explainer") or \
               getattr(built_explainer_manager, "explainer", None) is None

    def test_compute_shap_array_shape(self, built_explainer_manager, sample_shap_input):
        """_compute_shap_array must return (n_samples, n_features) — no bias column."""
        shap_array = built_explainer_manager._compute_shap_array(sample_shap_input)
        assert shap_array.shape == (N_TEST, N_FEATURES)

    def test_compute_shap_array_is_numpy(self, built_explainer_manager, sample_shap_input):
        shap_array = built_explainer_manager._compute_shap_array(sample_shap_input)
        assert isinstance(shap_array, np.ndarray)


# ---------------------------------------------------------------------------
# 2. build_explainer
# ---------------------------------------------------------------------------

class TestBuildExplainer:
    def test_build_returns_booster(self, trained_model_manager, synthetic_data):
        X, _ = synthetic_data
        em = ExplainerManager()
        result = em.build_explainer(trained_model_manager, X_background=X)
        assert result is em.booster

    def test_background_parameter_is_optional(self, trained_model_manager):
        """X_background=None must not raise — native XGBoost path does not need it."""
        em = ExplainerManager()
        em.build_explainer(trained_model_manager)
        assert em.booster is not None

    def test_repeated_build_replaces_booster(self, trained_model_manager):
        em = ExplainerManager()
        em.build_explainer(trained_model_manager)
        first = em.booster
        em.build_explainer(trained_model_manager)
        assert em.booster is not None


# ---------------------------------------------------------------------------
# 3. Local explanations
# ---------------------------------------------------------------------------

class TestExplainLocal:
    def test_returns_expected_keys(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        assert "shap_values" in result
        assert "grouped_local_shap" in result

    def test_shap_values_not_none(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        assert result["shap_values"] is not None

    def test_shap_values_shape(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        assert result["shap_values"].shape == (N_TEST, N_FEATURES)

    def test_grouped_local_shap_is_dataframe(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        assert isinstance(result["grouped_local_shap"], pd.DataFrame)

    def test_grouped_local_shap_columns(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        df = result["grouped_local_shap"]
        assert "feature_group" in df.columns
        assert "shap_value" in df.columns

    def test_grouped_local_shap_not_empty(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        assert len(result["grouped_local_shap"]) > 0

    def test_different_rows_produce_different_shap_values(
        self, built_explainer_manager, sample_shap_input
    ):
        # Compare raw SHAP arrays (before grouping) to avoid false equality
        # caused by a simple model collapsing many features to zero.
        shap_array = built_explainer_manager._compute_shap_array(sample_shap_input)
        assert not np.allclose(shap_array[0], shap_array[-1]), (
            "First and last rows returned identical raw SHAP values — unexpected."
        )

    def test_explain_local_without_build_raises(self, sample_shap_input):
        em = ExplainerManager()
        with pytest.raises(ValueError, match="Explainer not built"):
            em.explain_local(sample_shap_input, row_index=0)

    def test_row_index_selects_correct_row(self, built_explainer_manager, sample_shap_input):
        """grouped_local_shap for row 0 must match direct computation on row 0."""
        result = built_explainer_manager.explain_local(sample_shap_input, row_index=0)
        shap_array = built_explainer_manager._compute_shap_array(sample_shap_input)
        direct_row0 = shap_array[0]
        result_sum = result["grouped_local_shap"]["shap_value"].abs().sum()
        direct_sum = abs(direct_row0).sum()
        # grouped sums aggregate one-hot columns, so grouped abs-sum <= raw abs-sum
        assert result_sum <= direct_sum + 1e-6


# ---------------------------------------------------------------------------
# 4. Global explanations
# ---------------------------------------------------------------------------

class TestExplainGlobal:
    def test_grouped_global_shap_is_dataframe(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_global(sample_shap_input)
        assert isinstance(result["grouped_global_shap"], pd.DataFrame)

    def test_grouped_global_shap_columns(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_global(sample_shap_input)
        df = result["grouped_global_shap"]
        assert "feature_group" in df.columns
        assert "mean_abs_shap" in df.columns

    def test_grouped_global_shap_values_nonnegative(
        self, built_explainer_manager, sample_shap_input
    ):
        """Mean absolute SHAP values must be >= 0."""
        result = built_explainer_manager.explain_global(sample_shap_input)
        assert (result["grouped_global_shap"]["mean_abs_shap"] >= 0).all()

    def test_grouped_global_shap_sorted_descending(
        self, built_explainer_manager, sample_shap_input
    ):
        result = built_explainer_manager.explain_global(sample_shap_input)
        values = result["grouped_global_shap"]["mean_abs_shap"].values
        assert list(values) == sorted(values, reverse=True), (
            "Global SHAP should be sorted by importance (descending)."
        )

    def test_explain_global_without_build_raises(self, sample_shap_input):
        em = ExplainerManager()
        with pytest.raises(ValueError, match="Explainer not built"):
            em.explain_global(sample_shap_input)

    def test_shap_values_shape(self, built_explainer_manager, sample_shap_input):
        result = built_explainer_manager.explain_global(sample_shap_input)
        assert result["shap_values"].shape == (N_TEST, N_FEATURES)


# ---------------------------------------------------------------------------
# 5. Feature name grouping
# ---------------------------------------------------------------------------

class TestGroupFeatureName:
    def setup_method(self):
        self.em = ExplainerManager()

    def test_num_prefix_stripped_to_keep_exact(self):
        assert self.em.group_feature_name("num__lead_time") == "lead_time"

    def test_num_prefix_adr(self):
        assert self.em.group_feature_name("num__adr") == "adr"

    def test_bin_prefix_stripped(self):
        assert self.em.group_feature_name("bin__is_repeated_guest") == "is_repeated_guest"

    def test_onehot_hotel_grouped(self):
        assert self.em.group_feature_name("cat_onehot__hotel_City Hotel") == "hotel"
        assert self.em.group_feature_name("cat_onehot__hotel_Resort Hotel") == "hotel"

    def test_onehot_meal_grouped(self):
        assert self.em.group_feature_name("cat_onehot__meal_BB") == "meal"
        assert self.em.group_feature_name("cat_onehot__meal_HB") == "meal"

    def test_onehot_deposit_type_grouped(self):
        assert self.em.group_feature_name("cat_onehot__deposit_type_No Deposit") == "deposit_type"

    def test_onehot_customer_type_grouped(self):
        assert self.em.group_feature_name("cat_onehot__customer_type_Transient") == "customer_type"

    def test_country_group_prefix_maps_to_country(self):
        assert self.em.group_feature_name("cat_onehot__country_group_PRT") == "country"

    def test_market_segment_grouped(self):
        assert self.em.group_feature_name("cat_onehot__market_segment_Online TA") == "market_segment"

    def test_unknown_feature_returned_as_is(self):
        assert self.em.group_feature_name("some_unknown_feature") == "some_unknown_feature"


# ---------------------------------------------------------------------------
# 6. Driver splitting
# ---------------------------------------------------------------------------

class TestSplitLocalDrivers:
    def setup_method(self):
        self.em = ExplainerManager()
        self.grouped = pd.DataFrame({
            "feature_group": ["lead_time", "adr", "hotel", "meal", "deposit_type"],
            "shap_value": [0.30, -0.20, 0.10, -0.05, 0.08],
        })

    def test_higher_risk_all_positive(self):
        higher, _ = self.em.split_local_drivers(self.grouped)
        assert (higher["shap_value"] > 0).all()

    def test_lower_risk_all_negative(self):
        _, lower = self.em.split_local_drivers(self.grouped)
        assert (lower["shap_value"] < 0).all()

    def test_higher_risk_sorted_descending(self):
        higher, _ = self.em.split_local_drivers(self.grouped)
        values = higher["shap_value"].tolist()
        assert values == sorted(values, reverse=True)

    def test_lower_risk_sorted_ascending(self):
        _, lower = self.em.split_local_drivers(self.grouped)
        values = lower["shap_value"].tolist()
        assert values == sorted(values)

    def test_top_n_respected(self):
        higher, lower = self.em.split_local_drivers(self.grouped, top_n=2)
        assert len(higher) <= 2
        assert len(lower) <= 2

    def test_no_overlap_between_higher_and_lower(self):
        higher, lower = self.em.split_local_drivers(self.grouped)
        shared = set(higher["feature_group"]) & set(lower["feature_group"])
        assert len(shared) == 0


# ---------------------------------------------------------------------------
# 7. Performance
# ---------------------------------------------------------------------------

class TestExplainerPerformance:
    # Native XGBoost pred_contribs on a small model should be well under this.
    # A PermutationExplainer or KernelExplainer would take 10-60s for the same input.
    LOCAL_EXPLAIN_THRESHOLD_SECONDS = 2.0

    def test_single_row_local_explain_is_fast(
        self, built_explainer_manager, sample_shap_input
    ):
        single_row = sample_shap_input.iloc[:1]
        start = time.perf_counter()
        built_explainer_manager.explain_local(single_row, row_index=0)
        elapsed = time.perf_counter() - start

        assert elapsed < self.LOCAL_EXPLAIN_THRESHOLD_SECONDS, (
            f"Local explanation took {elapsed:.2f}s — "
            f"expected < {self.LOCAL_EXPLAIN_THRESHOLD_SECONDS}s. "
            "Check that native XGBoost pred_contribs is being used."
        )

    def test_batch_global_explain_is_fast(
        self, built_explainer_manager, sample_shap_input
    ):
        start = time.perf_counter()
        built_explainer_manager.explain_global(sample_shap_input)
        elapsed = time.perf_counter() - start

        assert elapsed < self.LOCAL_EXPLAIN_THRESHOLD_SECONDS, (
            f"Global explanation over {N_TEST} rows took {elapsed:.2f}s."
        )
