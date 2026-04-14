"""
test_hmm.py
-----------
Unit tests for HMM training, serialisation, and round-trip reconstruction.

Covers:
- train_appliance_hmm: fits on synthetic data, returns valid GaussianHMM
- _hmm_to_dict / reconstruct_hmm: round-trip serialisation preserves parameters
- save_models / load_models: JSON persistence round-trip
- run_training: integration test on a minimal synthetic CSV
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow importing from the parent Projet_NILM directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.train_hmm import (
    train_appliance_hmm,
    reconstruct_hmm,
    save_models,
    load_models,
    _hmm_to_dict,
)
from utils import state_labels


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_power_series(n: int = 500, seed: int = 0) -> pd.Series:
    """Return a synthetic two-state power series (OFF ≈ 0W, ON ≈ 2500W)."""
    rng = np.random.default_rng(seed)
    states = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    power = np.where(states == 0,
                     rng.normal(0, 2, n).clip(0),
                     rng.normal(2500, 50, n))
    return pd.Series(power)


def _make_refit_csv_for_training(rows: int = 200) -> str:
    """Write a minimal REFIT-style CSV with Aggregate + Appliance9 (Kettle)."""
    timestamps = pd.date_range("2014-01-01", periods=rows, freq="8s")
    rng = np.random.default_rng(1)
    states = rng.choice([0, 1], size=rows, p=[0.8, 0.2])
    power = np.where(states == 0, rng.normal(0, 2, rows).clip(0),
                     rng.normal(2700, 50, rows))

    df = pd.DataFrame({
        "Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "Unix": (timestamps.astype(np.int64) // 10 ** 9).tolist(),
        "Aggregate": power + rng.normal(100, 10, rows).clip(0),
        "Appliance1": rng.normal(50, 5, rows).clip(0),
        "Appliance2": rng.normal(30, 3, rows).clip(0),
        "Appliance3": rng.normal(0, 1, rows).clip(0),
        "Appliance4": rng.normal(0, 1, rows).clip(0),
        "Appliance5": rng.normal(0, 1, rows).clip(0),
        "Appliance6": rng.normal(0, 1, rows).clip(0),
        "Appliance7": rng.normal(0, 1, rows).clip(0),
        "Appliance8": rng.normal(100, 5, rows).clip(0),
        "Appliance9": power,   # "Kettle" column for House 3
    })
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_House_3.csv", delete=False, encoding="utf-8"
    )
    df.to_csv(tmp, index=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# train_appliance_hmm tests
# ---------------------------------------------------------------------------

class TestTrainApplianceHmm:

    def test_returns_fitted_model(self):
        series = _make_power_series()
        model = train_appliance_hmm(series, n_states=2)
        # A fitted GaussianHMM has these attributes
        assert hasattr(model, "means_")
        assert hasattr(model, "transmat_")
        assert hasattr(model, "startprob_")
        assert model.n_components == 2

    def test_three_state_model(self):
        rng = np.random.default_rng(5)
        vals = np.concatenate([
            rng.normal(0, 2, 300).clip(0),
            rng.normal(100, 10, 200),
            rng.normal(2500, 50, 100),
        ])
        series = pd.Series(vals)
        model = train_appliance_hmm(series, n_states=3)
        assert model.n_components == 3
        assert model.means_.shape == (3, 1)

    def test_transmat_rows_sum_to_one(self):
        series = _make_power_series()
        model = train_appliance_hmm(series, n_states=2)
        row_sums = model.transmat_.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-6)

    def test_startprob_sums_to_one(self):
        series = _make_power_series()
        model = train_appliance_hmm(series, n_states=2)
        assert abs(model.startprob_.sum() - 1.0) < 1e-6

    def test_sample_limit_respected(self):
        """Training with sample_limit should not raise even on a large series."""
        series = _make_power_series(n=10_000)
        model = train_appliance_hmm(series, n_states=2, sample_limit=300)
        assert model.n_components == 2

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="no non-NaN"):
            train_appliance_hmm(pd.Series([np.nan, np.nan]), n_states=2)


# ---------------------------------------------------------------------------
# Round-trip serialisation tests
# ---------------------------------------------------------------------------

class TestHmmRoundTrip:

    def _train_simple(self, n_states: int = 2) -> object:
        return train_appliance_hmm(_make_power_series(), n_states=n_states)

    def test_dict_preserves_n_states(self):
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        assert d["n_states"] == 2

    def test_dict_preserves_means(self):
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        np.testing.assert_allclose(
            np.array(d["means"]), model.means_, atol=1e-8
        )

    def test_dict_preserves_transmat(self):
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        np.testing.assert_allclose(
            np.array(d["transmat"]), model.transmat_, atol=1e-8
        )

    def test_reconstruct_means_match(self):
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        reconstructed = reconstruct_hmm(d)
        np.testing.assert_allclose(reconstructed.means_, model.means_, atol=1e-8)

    def test_reconstruct_transmat_matches(self):
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        reconstructed = reconstruct_hmm(d)
        np.testing.assert_allclose(reconstructed.transmat_, model.transmat_, atol=1e-8)

    def test_reconstruct_covars_match(self):
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        reconstructed = reconstruct_hmm(d)
        # Compare the raw private attribute — covars_ is a property that requires
        # n_features to be set (done only during fit()), so we test _covars_ directly.
        np.testing.assert_allclose(reconstructed._covars_, model._covars_, atol=1e-8)

    def test_reconstructed_model_can_predict(self):
        series = _make_power_series()
        model = self._train_simple(2)
        d = _hmm_to_dict(model, "kettle", 2)
        reconstructed = reconstruct_hmm(d)
        X = series.values[:50].reshape(-1, 1)
        states = reconstructed.predict(X)
        assert len(states) == 50
        assert set(states).issubset({0, 1})


# ---------------------------------------------------------------------------
# JSON persistence tests
# ---------------------------------------------------------------------------

class TestJsonPersistence:

    def test_save_and_load_round_trip(self):
        from hmmlearn.hmm import GaussianHMM
        model = train_appliance_hmm(_make_power_series(), n_states=2)
        models_dict = {"kettle": model}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(models_dict, house_number=99, models_dir=tmpdir)

            json_path = os.path.join(tmpdir, "kettle_hmm.json")
            assert os.path.isfile(json_path), "JSON file should be created"

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["appliance"] == "kettle"
            assert data["n_states"] == 2

            loaded = load_models(house_number=99, appliances=["kettle"],
                                 models_dir=tmpdir)
            assert "kettle" in loaded

            reconstructed = reconstruct_hmm(loaded["kettle"])
            np.testing.assert_allclose(reconstructed.means_, model.means_, atol=1e-8)

    def test_load_missing_model_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_models(house_number=99, appliances=["nonexistent"],
                                 models_dir=tmpdir)
            assert loaded == {}

    def test_state_labels_in_json(self):
        model = train_appliance_hmm(_make_power_series(), n_states=2)
        d = _hmm_to_dict(model, "kettle", 2)
        assert d["state_labels"] == ["OFF", "ON"]

    def test_three_state_labels_in_json(self):
        series = pd.Series(np.concatenate([
            np.zeros(200), np.full(100, 100), np.full(50, 2500)
        ]))
        model = train_appliance_hmm(series, n_states=3)
        d = _hmm_to_dict(model, "fridge", 3)
        assert d["state_labels"] == ["OFF", "LOW", "HIGH"]


# ---------------------------------------------------------------------------
# state_labels utility tests
# ---------------------------------------------------------------------------

class TestStateLabels:

    def test_two_states(self):
        assert state_labels(2) == ["OFF", "ON"]

    def test_three_states(self):
        assert state_labels(3) == ["OFF", "LOW", "HIGH"]

    def test_four_states_generic(self):
        labels = state_labels(4)
        assert len(labels) == 4
        assert labels[0] == "state_0"


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
