"""
test_realtime.py
----------------
Unit tests for the real-time NILM inference engine
(:class:`pipeline.realtime.RealTimeNILM`).

Test categories
~~~~~~~~~~~~~~~
- ``TestSmoothingHelper``    – ``_smooth()`` correctness for both methods.
- ``TestBufferWarmUp``       – ``is_warmed_up`` transitions correctly.
- ``TestSmoothing``          – smoothed_power tracks the input buffer.
- ``TestHysteresis``         – committed state ignores sub-threshold changes.
- ``TestTransitionPenalty``  – penalty discourages per-step combo jumps.
- ``TestStateStabilisation`` – majority vote over the state buffer.
- ``TestOutputSchema``       – every required key is present and typed correctly.
- ``TestStableTransitions``  – plausible state changes are eventually accepted.
- ``TestReset``              – reset clears all internal state.
- ``TestReprStr``            – __repr__ is informative.

All tests use synthetic (in-memory) HMM models so no real model files are
needed.  The synthetic models are constructed to have unambiguous, well-
separated power states so that inference outcomes are deterministic.
"""

from __future__ import annotations

import os
import sys
import tempfile
import json

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup – allow importing from the Projet_NILM root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.realtime import RealTimeNILM, _smooth
from collections import deque


# ---------------------------------------------------------------------------
# Synthetic model helpers
# ---------------------------------------------------------------------------

def _make_model_dict(appliance: str, means: list[float]) -> dict:
    """Build a minimal HMM model dict with well-separated, zero-variance states.

    Uses a diagonal transition matrix (stays in current state), which makes
    single-step predictions deterministic given the current state.
    """
    n = len(means)
    labels_map = {2: ["OFF", "ON"], 3: ["OFF", "LOW", "HIGH"]}
    labels = labels_map.get(n, [f"state_{i}" for i in range(n)])

    # Sort means ascending so that semantic labels align correctly.
    sorted_means = sorted(means)
    # Near-identity transition matrix (98 % self-loop)
    transmat = np.eye(n) * 0.98 + np.full((n, n), 0.02 / n)
    transmat /= transmat.sum(axis=1, keepdims=True)

    return {
        "appliance": appliance,
        "n_states": n,
        "state_labels": labels,
        "startprob": [1.0 / n] * n,
        "transmat": transmat.tolist(),
        "means": [[m] for m in sorted_means],
        # Very tight covariance → near-zero variance, deterministic emission
        "covars": [[[1e-4]] for _ in range(n)],
    }


def _write_models(models: dict[str, dict], tmpdir: str) -> None:
    """Serialise model dicts to JSON files under *tmpdir*."""
    for appliance, model_dict in models.items():
        path = os.path.join(tmpdir, f"{appliance}_hmm.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_dict, f)


def _build_rt(
    appliances: list[str],
    means_map: dict[str, list[float]],
    input_buffer_size: int = 5,
    state_buffer_size: int = 5,
    hysteresis_margin: float = 10.0,
    transition_penalty: float = 20.0,
) -> tuple[RealTimeNILM, str]:
    """Return a (RealTimeNILM, tmpdir) tuple using synthetic models.

    The caller is responsible for cleaning up *tmpdir*.
    """
    tmpdir = tempfile.mkdtemp()
    models = {app: _make_model_dict(app, means_map[app]) for app in appliances}
    _write_models(models, tmpdir)

    rt = RealTimeNILM(
        house_number=99,
        appliances=appliances,
        models_dir=tmpdir,
        input_buffer_size=input_buffer_size,
        state_buffer_size=state_buffer_size,
        hysteresis_margin=hysteresis_margin,
        transition_penalty=transition_penalty,
    )
    return rt, tmpdir


# ---------------------------------------------------------------------------
# _smooth helper tests
# ---------------------------------------------------------------------------

class TestSmoothingHelper:
    """Unit tests for the standalone _smooth() function."""

    def test_median_single_value(self):
        buf = deque([100.0])
        assert _smooth(buf, "median") == pytest.approx(100.0)

    def test_median_odd_count(self):
        buf = deque([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _smooth(buf, "median") == pytest.approx(30.0)

    def test_median_robust_to_spike(self):
        buf = deque([100.0, 100.0, 100.0, 100.0, 9999.0])
        assert _smooth(buf, "median") == pytest.approx(100.0)

    def test_trimmed_mean_no_outliers(self):
        buf = deque([100.0, 110.0, 105.0, 102.0, 108.0])
        result = _smooth(buf, "trimmed_mean")
        # Should be close to the plain mean since no strong outliers
        assert 100.0 < result < 115.0

    def test_trimmed_mean_extreme_outlier_clipped(self):
        buf = deque([100.0, 100.0, 100.0, 100.0, 5000.0])
        result_trimmed = _smooth(buf, "trimmed_mean")
        result_plain_mean = float(np.mean([100.0, 100.0, 100.0, 100.0, 5000.0]))
        # Trimmed mean should be closer to 100 than the plain mean
        assert result_trimmed < result_plain_mean

    def test_unknown_method_falls_back_to_median(self):
        buf = deque([10.0, 20.0, 30.0])
        # Any unknown method string falls back to median
        assert _smooth(buf, "unknown_method") == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Buffer warm-up tests
# ---------------------------------------------------------------------------

class TestBufferWarmUp:
    """Verify is_warmed_up transitions at the correct sample count."""

    def setup_method(self):
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=4,
            state_buffer_size=6,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_not_warmed_up_initially(self):
        assert not self.rt.is_warmed_up

    def test_not_warmed_up_after_partial_fill(self):
        for _ in range(3):
            self.rt.update(0.0)
        assert not self.rt.is_warmed_up

    def test_warmed_up_after_max_buffer_samples(self):
        # max(input_buffer=4, state_buffer=6) = 6 samples needed
        for _ in range(6):
            self.rt.update(0.0)
        assert self.rt.is_warmed_up

    def test_result_has_warmed_up_key(self):
        result = self.rt.update(0.0)
        assert "warmed_up" in result


# ---------------------------------------------------------------------------
# Smoothing behaviour tests
# ---------------------------------------------------------------------------

class TestSmoothing:
    """Smoothed power should track the rolling median of recent values."""

    def setup_method(self):
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=5,
            state_buffer_size=3,
            hysteresis_margin=0.0,
            transition_penalty=0.0,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_smoothed_equals_single_value_on_first_call(self):
        result = self.rt.update(300.0)
        assert result["smoothed_power"] == pytest.approx(300.0)

    def test_spike_does_not_dominate_median_smoothing(self):
        # Fill buffer with a stable value, then send a spike.
        for _ in range(4):
            self.rt.update(200.0)
        result = self.rt.update(9000.0)
        # Median of [200, 200, 200, 200, 9000] = 200
        assert result["smoothed_power"] == pytest.approx(200.0)

    def test_smoothed_power_key_present(self):
        result = self.rt.update(500.0)
        assert "smoothed_power" in result
        assert isinstance(result["smoothed_power"], float)

    def test_raw_power_preserved_exactly(self):
        result = self.rt.update(1234.56)
        assert result["raw_power"] == pytest.approx(1234.56)


# ---------------------------------------------------------------------------
# Hysteresis tests
# ---------------------------------------------------------------------------

class TestHysteresis:
    """The committed state must not change unless the improvement exceeds the margin."""

    def setup_method(self):
        # kettle: OFF=0W, ON=2000W — well-separated
        # large hysteresis: 500W margin, no transition penalty
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=1,   # no smoothing lag
            state_buffer_size=1,   # no vote lag
            hysteresis_margin=500.0,
            transition_penalty=0.0,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_small_power_change_does_not_flip_state(self):
        # Anchor to OFF (cost |100-0|=100 for OFF combo vs |100-2000|=1900 for ON)
        self.rt.update(100.0)

        # Nudge toward ON but cost improvement < 500W margin:
        # OFF cost = |450-0| = 450, ON cost = |450-2000| = 1550.  Improvement = -1100
        # → stays OFF
        result = self.rt.update(450.0)
        assert result["appliances"]["kettle"]["state_label"] == "OFF"

    def test_large_power_change_flips_state(self):
        # Anchor to OFF with a clear OFF reading
        self.rt.update(0.0)
        # Now clearly ON: OFF cost=2000, ON cost≈0 → improvement 2000 > 500
        result = self.rt.update(2000.0)
        assert result["appliances"]["kettle"]["state_label"] == "ON"

    def test_oscillation_suppressed_near_midpoint(self):
        """Rapid alternation near the midpoint should be dampened by hysteresis."""
        self.rt.update(0.0)   # anchor OFF
        labels = []
        for _ in range(10):
            r = self.rt.update(800.0)   # midpoint — OFF cost=800, ON cost=1200
            labels.append(r["appliances"]["kettle"]["state_label"])
        # With large hysteresis all should remain OFF (anchored)
        assert all(lbl == "OFF" for lbl in labels)


# ---------------------------------------------------------------------------
# State-stabilisation (majority vote) tests
# ---------------------------------------------------------------------------

class TestStateStabilisation:
    """Majority vote over the state buffer should suppress single-sample flicker."""

    def setup_method(self):
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=1,
            state_buffer_size=7,
            hysteresis_margin=0.0,
            transition_penalty=0.0,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_single_spike_does_not_change_stable_output(self):
        # Feed 6 OFF samples to fill most of the buffer
        for _ in range(6):
            self.rt.update(0.0)
        # One ON spike: buffer is [OFF, OFF, OFF, OFF, OFF, OFF, ON]
        result = self.rt.update(2000.0)
        # 6 votes for OFF vs 1 for ON → majority is still OFF
        assert result["appliances"]["kettle"]["state_label"] == "OFF"

    def test_consistent_on_signal_produces_on_after_buffer_filled(self):
        # Feed ON samples until buffer is full
        for _ in range(7):
            self.rt.update(2000.0)
        result = self.rt.update(2000.0)
        assert result["appliances"]["kettle"]["state_label"] == "ON"

    def test_confidence_is_one_for_unanimous_buffer(self):
        for _ in range(7):
            self.rt.update(0.0)
        result = self.rt.update(0.0)
        assert result["appliances"]["kettle"]["confidence"] == pytest.approx(1.0)

    def test_confidence_below_one_for_mixed_buffer(self):
        for _ in range(4):
            self.rt.update(0.0)
        for _ in range(3):
            self.rt.update(2000.0)
        result = self.rt.update(0.0)
        assert result["appliances"]["kettle"]["confidence"] < 1.0


# ---------------------------------------------------------------------------
# Transition penalty tests
# ---------------------------------------------------------------------------

class TestTransitionPenalty:
    """Transition penalty should keep the committed combo stable for ambiguous inputs."""

    def setup_method(self):
        # kettle: OFF=0W, ON=2000W
        # No hysteresis, but large transition penalty
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=1,
            state_buffer_size=1,
            hysteresis_margin=0.0,
            transition_penalty=2000.0,  # extremely large penalty
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_penalty_keeps_committed_state_under_ambiguous_input(self):
        # Anchor to OFF
        self.rt.update(0.0)
        # Send midpoint value: normally best combo alternates,
        # but penalty should keep it at OFF.
        # OFF cost = |1000-0| = 1000; ON (penalised) = |1000-2000| + 2000 = 3000
        result = self.rt.update(1000.0)
        assert result["appliances"]["kettle"]["state_label"] == "OFF"

    def test_penalty_zero_allows_free_switching(self):
        rt2, tmpdir2 = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=1,
            state_buffer_size=1,
            hysteresis_margin=0.0,
            transition_penalty=0.0,
        )
        try:
            rt2.update(0.0)   # anchor OFF
            # With no penalty, 2000W clearly maps to ON regardless of anchor
            result = rt2.update(2000.0)
            assert result["appliances"]["kettle"]["state_label"] == "ON"
        finally:
            import shutil
            shutil.rmtree(tmpdir2, ignore_errors=True)


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Every required key must be present with the correct type."""

    def setup_method(self):
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle", "fridge"],
            means_map={"kettle": [0.0, 2000.0], "fridge": [0.0, 50.0, 250.0]},
            input_buffer_size=3,
            state_buffer_size=3,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_top_level_keys_present(self):
        result = self.rt.update(200.0)
        for key in ("timestamp", "raw_power", "smoothed_power", "warmed_up", "appliances"):
            assert key in result, f"Missing top-level key: {key!r}"

    def test_timestamp_is_string(self):
        result = self.rt.update(200.0)
        assert isinstance(result["timestamp"], str)

    def test_raw_power_is_float(self):
        result = self.rt.update(200.0)
        assert isinstance(result["raw_power"], float)

    def test_smoothed_power_is_float(self):
        result = self.rt.update(200.0)
        assert isinstance(result["smoothed_power"], float)

    def test_warmed_up_is_bool(self):
        result = self.rt.update(200.0)
        assert isinstance(result["warmed_up"], bool)

    def test_appliances_has_all_requested_keys(self):
        result = self.rt.update(200.0)
        for app in ["kettle", "fridge"]:
            assert app in result["appliances"], f"Missing appliance: {app!r}"

    def test_per_appliance_keys_present(self):
        result = self.rt.update(200.0)
        for app in result["appliances"]:
            info = result["appliances"][app]
            for key in ("state_label", "state_index", "estimated_power", "confidence"):
                assert key in info, f"'{app}' missing key {key!r}"

    def test_per_appliance_types(self):
        result = self.rt.update(200.0)
        for app, info in result["appliances"].items():
            assert isinstance(info["state_label"], str), f"{app}: state_label not str"
            assert isinstance(info["state_index"], int), f"{app}: state_index not int"
            assert isinstance(info["estimated_power"], float), f"{app}: estimated_power not float"
            assert isinstance(info["confidence"], float), f"{app}: confidence not float"

    def test_confidence_in_range(self):
        result = self.rt.update(200.0)
        for app, info in result["appliances"].items():
            assert 0.0 <= info["confidence"] <= 1.0, (
                f"{app}: confidence {info['confidence']} out of [0,1]"
            )

    def test_state_label_is_semantic(self):
        result = self.rt.update(0.0)
        assert result["appliances"]["kettle"]["state_label"] in ("OFF", "ON")
        assert result["appliances"]["fridge"]["state_label"] in ("OFF", "LOW", "HIGH")

    def test_custom_timestamp_is_preserved(self):
        ts = "2024-06-01T12:00:00+00:00"
        result = self.rt.update(100.0, timestamp=ts)
        assert result["timestamp"] == ts


# ---------------------------------------------------------------------------
# Stable state transition tests
# ---------------------------------------------------------------------------

class TestStableTransitions:
    """Sustained power changes must eventually be reflected in the output."""

    def setup_method(self):
        # Both buffers are small so transitions propagate quickly.
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=3,
            state_buffer_size=3,
            hysteresis_margin=50.0,
            transition_penalty=30.0,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_transition_off_to_on_detected(self):
        # Establish stable OFF
        for _ in range(5):
            self.rt.update(0.0)

        # Sustained ON signal — should flip within buffer_size steps
        final = None
        for _ in range(10):
            final = self.rt.update(2000.0)
        assert final["appliances"]["kettle"]["state_label"] == "ON"

    def test_transition_on_to_off_detected(self):
        # Establish stable ON
        for _ in range(5):
            self.rt.update(2000.0)

        # Sustained OFF signal
        final = None
        for _ in range(10):
            final = self.rt.update(0.0)
        assert final["appliances"]["kettle"]["state_label"] == "OFF"

    def test_predicted_power_matches_state(self):
        for _ in range(10):
            result = self.rt.update(2000.0)
        info = result["appliances"]["kettle"]
        assert info["state_label"] == "ON"
        assert info["estimated_power"] == pytest.approx(2000.0, abs=1.0)


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    """reset() must restore the engine to its initial (empty-buffer) state."""

    def setup_method(self):
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
            input_buffer_size=5,
            state_buffer_size=5,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_not_warmed_up_after_reset(self):
        for _ in range(10):
            self.rt.update(0.0)
        assert self.rt.is_warmed_up
        self.rt.reset()
        assert not self.rt.is_warmed_up

    def test_buffers_empty_after_reset(self):
        for _ in range(10):
            self.rt.update(0.0)
        self.rt.reset()
        assert len(self.rt._input_buffer) == 0
        for buf in self.rt._state_buffers.values():
            assert len(buf) == 0

    def test_inference_continues_after_reset(self):
        for _ in range(5):
            self.rt.update(0.0)
        self.rt.reset()
        result = self.rt.update(2000.0)
        assert "appliances" in result


# ---------------------------------------------------------------------------
# __repr__ test
# ---------------------------------------------------------------------------

class TestReprStr:

    def test_repr_contains_house_number(self):
        rt, tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
        )
        try:
            assert "99" in repr(rt)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_repr_contains_appliance_name(self):
        rt, tmpdir = _build_rt(
            appliances=["kettle"],
            means_map={"kettle": [0.0, 2000.0]},
        )
        try:
            assert "kettle" in repr(rt)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Multi-appliance interaction tests
# ---------------------------------------------------------------------------

class TestMultiAppliance:
    """Ensure inference works correctly when multiple appliances are active."""

    def setup_method(self):
        # kettle: 0W or 2000W  /  fridge: 0W, 50W, 250W
        self.rt, self.tmpdir = _build_rt(
            appliances=["kettle", "fridge"],
            means_map={"kettle": [0.0, 2000.0], "fridge": [0.0, 50.0, 250.0]},
            input_buffer_size=3,
            state_buffer_size=3,
            hysteresis_margin=30.0,
            transition_penalty=30.0,
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_both_appliances_in_output(self):
        result = self.rt.update(200.0)
        assert "kettle" in result["appliances"]
        assert "fridge" in result["appliances"]

    def test_combined_power_kettle_and_fridge_high(self):
        """~2250W should map to kettle ON + fridge HIGH after buffer warm-up."""
        for _ in range(10):
            result = self.rt.update(2250.0)  # 2000 + 250
        assert result["appliances"]["kettle"]["state_label"] == "ON"
        assert result["appliances"]["fridge"]["state_label"] == "HIGH"

    def test_low_aggregate_maps_all_off(self):
        """~0W should map all appliances to OFF after buffer warm-up."""
        for _ in range(10):
            result = self.rt.update(0.0)
        for app in ("kettle", "fridge"):
            assert result["appliances"][app]["state_label"] == "OFF"


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
