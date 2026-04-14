"""
test_preprocessing.py
---------------------
Unit tests for preprocessing utilities.

Covers:
- hampel_filter: spike detection and replacement
- interpolate_missing: gap filling with max-gap limit
- load_refit_csv: column validation and timestamp parsing
- preprocess_house: end-to-end pipeline on a synthetic fixture
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow importing from the parent Projet_NILM directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessing import hampel_filter, interpolate_missing, load_refit_csv, preprocess_house


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_refit_csv(rows: int = 50, add_spike: bool = False) -> str:
    """Write a minimal REFIT-style CSV to a temp file and return its path.

    Returns a path to a temporary file that callers must delete.
    """
    timestamps = pd.date_range("2014-01-01", periods=rows, freq="8s")
    aggregate = np.full(rows, 200.0)
    appliance1 = np.full(rows, 0.5)

    if add_spike:
        # Insert obvious outlier spikes
        aggregate[10] = 99_000.0
        appliance1[20] = 50_000.0

    df = pd.DataFrame({
        "Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "Unix": (timestamps.astype(np.int64) // 10 ** 9).tolist(),
        "Aggregate": aggregate,
        "Appliance1": appliance1,
    })

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    df.to_csv(tmp, index=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# hampel_filter tests
# ---------------------------------------------------------------------------

class TestHampelFilter:

    def test_clean_series_unchanged(self):
        """A flat series with no outliers should be returned unchanged."""
        s = pd.Series(np.full(100, 200.0))
        filtered, mask = hampel_filter(s)
        assert mask.sum() == 0, "No outliers expected in a flat series"
        np.testing.assert_allclose(filtered.values, s.values, atol=1e-6)

    def test_spike_is_detected(self):
        """A large isolated spike should be flagged as an outlier."""
        s = pd.Series(np.full(60, 5.0))
        s[30] = 99_000.0
        _, mask = hampel_filter(s)
        assert mask.iloc[30], "Spike at index 30 should be detected"

    def test_spike_is_replaced(self):
        """A spike should be replaced by the local median, not the spike value."""
        s = pd.Series(np.full(60, 5.0))
        s[30] = 99_000.0
        filtered, _ = hampel_filter(s)
        assert filtered.iloc[30] < 100.0, (
            f"Spike should be replaced by ~5W, got {filtered.iloc[30]:.1f}"
        )

    def test_custom_window_and_sigma(self):
        """Custom window/sigma parameters should be accepted without error."""
        s = pd.Series(np.random.default_rng(0).normal(100, 5, 200))
        filtered, mask = hampel_filter(s, window_size=5, n_sigmas=2.0)
        assert len(filtered) == len(s)
        assert len(mask) == len(s)

    def test_output_same_length_as_input(self):
        s = pd.Series(np.arange(50, dtype=float))
        filtered, mask = hampel_filter(s)
        assert len(filtered) == 50
        assert len(mask) == 50


# ---------------------------------------------------------------------------
# interpolate_missing tests
# ---------------------------------------------------------------------------

class TestInterpolateMissing:

    def test_no_nans_unchanged(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = interpolate_missing(s)
        np.testing.assert_array_equal(result.values, s.values)

    def test_single_nan_filled(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = interpolate_missing(s)
        assert not result.isna().any(), "Single NaN should be interpolated"
        assert abs(result.iloc[1] - 2.0) < 1e-6, "Linear interpolation: 1→3 midpoint = 2"

    def test_gap_within_max_filled(self):
        """A run of NaNs shorter than max_gap should be fully filled."""
        s = pd.Series([0.0, np.nan, np.nan, np.nan, 4.0])
        result = interpolate_missing(s, max_gap=5)
        assert not result.isna().any()

    def test_gap_exceeding_max_left_as_nan(self):
        """A run of NaNs longer than max_gap should NOT be fully filled."""
        s = pd.Series([0.0] + [np.nan] * 10 + [10.0])
        result = interpolate_missing(s, max_gap=3)
        # Some NaNs in the long run should remain
        assert result.isna().any(), (
            "NaN run longer than max_gap should have remaining NaNs"
        )


# ---------------------------------------------------------------------------
# load_refit_csv tests
# ---------------------------------------------------------------------------

class TestLoadRefitCsv:

    def test_loads_synthetic_csv(self):
        path = _make_refit_csv(rows=30)
        try:
            df = load_refit_csv(path)
            assert len(df) == 30
            assert "Aggregate" in df.columns
            assert "Appliance1" in df.columns
            assert "Unix" not in df.columns, "Unix column should be dropped"
        finally:
            os.unlink(path)

    def test_max_rows_limits_output(self):
        path = _make_refit_csv(rows=100)
        try:
            df = load_refit_csv(path, max_rows=20)
            assert len(df) <= 20
        finally:
            os.unlink(path)

    def test_index_is_datetime(self):
        path = _make_refit_csv(rows=10)
        try:
            df = load_refit_csv(path)
            assert isinstance(df.index, pd.DatetimeIndex)
        finally:
            os.unlink(path)

    def test_negative_values_clipped_to_zero(self):
        """Negative power readings (sensor errors) should be clipped to 0."""
        timestamps = pd.date_range("2014-01-01", periods=5, freq="8s")
        df_raw = pd.DataFrame({
            "Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "Aggregate": [-10.0, 100.0, -5.0, 200.0, 0.0],
        })
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        df_raw.to_csv(tmp, index=False)
        tmp.close()
        try:
            df = load_refit_csv(tmp.name)
            assert (df["Aggregate"] >= 0).all(), "Negative values should be clipped to 0"
        finally:
            os.unlink(tmp.name)

    def test_missing_aggregate_raises(self):
        """CSV without an 'Aggregate' column should raise ValueError."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        tmp.write("Time,Appliance1\n2014-01-01,100\n")
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Aggregate"):
                load_refit_csv(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_refit_csv("/nonexistent/path/house_99.csv")

    def test_spike_survives_load(self):
        """load_refit_csv should NOT apply Hampel — that is preprocess_house's job."""
        path = _make_refit_csv(rows=50, add_spike=True)
        try:
            df = load_refit_csv(path)
            assert df["Aggregate"].max() > 1000, (
                "Spike should still be present after raw CSV load"
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# preprocess_house end-to-end test
# ---------------------------------------------------------------------------

class TestPreprocessHouse:

    def test_full_pipeline_removes_spikes(self):
        """preprocess_house should remove the synthetic spike via Hampel filter."""
        path = _make_refit_csv(rows=80, add_spike=True)
        try:
            df = preprocess_house(path)
            assert df["Aggregate"].max() < 1000, (
                f"Spike should be removed; max = {df['Aggregate'].max():.0f}"
            )
        finally:
            os.unlink(path)

    def test_output_has_no_nans(self):
        path = _make_refit_csv(rows=50)
        try:
            df = preprocess_house(path)
            assert df.isna().sum().sum() == 0, "No NaN values should remain after preprocessing"
        finally:
            os.unlink(path)

    def test_output_non_negative(self):
        path = _make_refit_csv(rows=50)
        try:
            df = preprocess_house(path)
            assert (df >= 0).all().all(), "All values should be non-negative"
        finally:
            os.unlink(path)

    def test_datetime_index(self):
        path = _make_refit_csv(rows=30)
        try:
            df = preprocess_house(path)
            assert isinstance(df.index, pd.DatetimeIndex)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
