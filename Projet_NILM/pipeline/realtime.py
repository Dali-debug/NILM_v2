"""
realtime.py
-----------
Real-time NILM inference engine using pre-trained Gaussian HMM models.

Accepts one aggregate power sample at a time and returns per-appliance state
estimates with built-in input smoothing, state stabilisation via majority
voting, transition-penalty-based temporal continuity, and hysteresis.

Architecture
~~~~~~~~~~~~
Each call to :meth:`RealTimeNILM.update` performs the following steps:

1. **Input smoothing** – the raw aggregate value is appended to a fixed-size
   :class:`collections.deque`.  A median (or trimmed-mean) of the buffer is
   computed to obtain a noise-robust ``smoothed_power``.

2. **Combinatorial search with transition penalty** – the smoothed power is
   compared against the predicted total of every pre-computed state
   combination.  A constant ``transition_penalty`` (Watts) is added to the
   cost of every combination that differs from the currently committed one,
   biasing the search toward temporal continuity.

3. **Hysteresis** – the newly selected combination replaces the committed one
   only when its unpenalised cost beats the committed combination's cost by
   more than ``hysteresis_margin`` Watts.  This creates a dead zone that
   prevents rapid oscillation near decision boundaries.

4. **State stabilisation** – the raw (per-step) state index for each
   appliance is appended to a per-appliance fixed-size
   :class:`collections.deque`.  The majority-voted state (mode of the buffer)
   is used as the stable output, preventing single-sample flicker.

5. **Structured output** – a :class:`dict` is returned with timestamps, raw
   and smoothed power, and per-appliance ``state_label``, ``state_index``,
   ``estimated_power``, and ``confidence`` (vote fraction).

All tunable constants have defaults sourced from ``config`` so they can be
changed from a single location without touching this module.

Usage::

    from pipeline.realtime import RealTimeNILM

    rt = RealTimeNILM(house_number=3)
    result = rt.update(aggregate_power=1500.0)
    # result["appliances"]["kettle"]["state_label"] → "OFF" / "ON"
"""

from __future__ import annotations

import itertools
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config import (
    DEFAULT_APPLIANCES,
    RT_HYSTERESIS_MARGIN,
    RT_INPUT_BUFFER_SIZE,
    RT_SMOOTHING_METHOD,
    RT_STATE_BUFFER_SIZE,
    RT_TRANSITION_PENALTY,
)
from pipeline.disaggregate import _semantic_state_label_map
from pipeline.train_hmm import load_models, reconstruct_hmm
from utils import get_logger, state_labels

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

#: Dict returned by :meth:`RealTimeNILM.update`.
InferenceResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Helper: smoothing
# ---------------------------------------------------------------------------

def _smooth(buffer: deque, method: str) -> float:
    """Return a robust estimate of the central value in *buffer*.

    Parameters
    ----------
    buffer : deque
        Recent aggregate power readings (Watts).
    method : str
        ``"median"`` or ``"trimmed_mean"``.

    Returns
    -------
    float
        Smoothed power value.
    """
    arr = np.asarray(buffer, dtype=float)
    if method == "trimmed_mean":
        q25, q75 = np.percentile(arr, [25.0, 75.0])
        iqr = q75 - q25
        lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(arr))
    # Default: median
    return float(np.median(arr))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RealTimeNILM:
    """Real-time NILM inference for a fixed set of appliances.

    Parameters
    ----------
    house_number : int
        House number whose trained models will be loaded.
    appliances : list of str, optional
        Canonical appliance names to include.
        Defaults to ``config.DEFAULT_APPLIANCES``.
    models_dir : str, optional
        Directory containing ``<appliance>_hmm.json`` files.
        Defaults to ``models/<house_number>/`` relative to this file.
    input_buffer_size : int
        Length of the rolling window used for aggregate smoothing.
        Defaults to ``config.RT_INPUT_BUFFER_SIZE``.
    state_buffer_size : int
        Length of the per-appliance state history used for majority voting.
        Defaults to ``config.RT_STATE_BUFFER_SIZE``.
    smoothing_method : str
        ``"median"`` or ``"trimmed_mean"``.
        Defaults to ``config.RT_SMOOTHING_METHOD``.
    hysteresis_margin : float
        Minimum improvement in Watts a new combination must achieve over the
        committed one before a state change is accepted.
        Defaults to ``config.RT_HYSTERESIS_MARGIN``.
    transition_penalty : float
        Extra cost (Watts) applied to non-committed combinations during
        the combinatorial search to enforce temporal continuity.
        Defaults to ``config.RT_TRANSITION_PENALTY``.

    Attributes
    ----------
    appliances : list of str
        Ordered list of appliance names being tracked.
    n_combinations : int
        Total number of pre-computed state combinations.
    is_warmed_up : bool
        ``True`` once both the input buffer and all state buffers are full.

    Examples
    --------
    >>> rt = RealTimeNILM(house_number=3)
    >>> result = rt.update(1500.0)
    >>> result["appliances"]["kettle"]["state_label"]
    'OFF'
    """

    def __init__(
        self,
        house_number: int,
        appliances: list[str] | None = None,
        models_dir: str | None = None,
        input_buffer_size: int = RT_INPUT_BUFFER_SIZE,
        state_buffer_size: int = RT_STATE_BUFFER_SIZE,
        smoothing_method: str = RT_SMOOTHING_METHOD,
        hysteresis_margin: float = RT_HYSTERESIS_MARGIN,
        transition_penalty: float = RT_TRANSITION_PENALTY,
    ) -> None:
        if appliances is None:
            appliances = list(DEFAULT_APPLIANCES)

        self._house_number = house_number
        self._smoothing_method = smoothing_method
        self._hysteresis_margin = float(hysteresis_margin)
        self._transition_penalty = float(transition_penalty)

        # --- load and reconstruct HMM models ---
        raw_models = load_models(house_number, appliances, models_dir=models_dir)
        if not raw_models:
            raise RuntimeError(
                f"No trained models found for house {house_number}. "
                "Run train_hmm.py first."
            )

        # Build per-appliance metadata in a stable order
        self._app_info: list[tuple] = []
        for name in appliances:
            if name not in raw_models:
                logger.warning("No model for '%s' — skipping.", name)
                continue
            model_dict = raw_models[name]
            hmm = reconstruct_hmm(model_dict)
            labels = model_dict.get("state_labels", state_labels(hmm.n_components))
            means = hmm.means_.flatten().tolist()          # indexed by raw HMM state
            idx_to_label = _semantic_state_label_map(hmm, labels)
            self._app_info.append((name, hmm, means, labels, idx_to_label))
            logger.debug(
                "Loaded '%s': %d states, means=%s",
                name, hmm.n_components, [round(m, 1) for m in means],
            )

        if not self._app_info:
            raise RuntimeError("None of the requested appliances have loaded models.")

        self.appliances: list[str] = [info[0] for info in self._app_info]

        # --- pre-compute state combinations once ---
        state_ranges = [range(len(info[2])) for info in self._app_info]
        self._combinations: list[tuple[int, ...]] = list(
            itertools.product(*state_ranges)
        )
        # (n_combos,) vector of predicted aggregate powers
        self._combo_powers: np.ndarray = np.array([
            sum(self._app_info[j][2][combo[j]] for j in range(len(self._app_info)))
            for combo in self._combinations
        ], dtype=float)

        self.n_combinations: int = len(self._combinations)
        logger.info(
            "RealTimeNILM ready | house=%d | appliances=%s | combinations=%d",
            house_number, self.appliances, self.n_combinations,
        )

        # --- buffers ---
        self._input_buffer: deque = deque(maxlen=input_buffer_size)
        self._state_buffers: dict[str, deque] = {
            name: deque(maxlen=state_buffer_size) for name in self.appliances
        }
        self._input_buffer_size = input_buffer_size
        self._state_buffer_size = state_buffer_size

        # --- inference state ---
        self._committed_combo_idx: int | None = None  # committed after hysteresis
        self._sample_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_warmed_up(self) -> bool:
        """``True`` once all internal buffers are full."""
        if len(self._input_buffer) < self._input_buffer_size:
            return False
        return all(
            len(self._state_buffers[name]) >= self._state_buffer_size
            for name in self.appliances
        )

    def update(
        self,
        aggregate_power: float,
        timestamp: str | None = None,
    ) -> InferenceResult:
        """Process one aggregate power sample and return per-appliance states.

        Parameters
        ----------
        aggregate_power : float
            Observed aggregate power in Watts.
        timestamp : str, optional
            ISO-8601 timestamp for this sample.  Defaults to the current
            UTC time.

        Returns
        -------
        dict
            Structured result with the following keys:

            ``timestamp`` : str
                ISO-8601 timestamp.
            ``raw_power`` : float
                The original unsmoothed aggregate value (Watts).
            ``smoothed_power`` : float
                Robust smoothed aggregate after the input buffer.
            ``warmed_up`` : bool
                Whether both buffers are fully populated.
            ``appliances`` : dict[str, dict]
                Per-appliance sub-dict with:

                - ``state_label``   : str  (e.g. ``"OFF"``, ``"ON"``, ``"HIGH"``)
                - ``state_index``   : int  (raw HMM state index after majority vote)
                - ``estimated_power`` : float  (mean Watts for this state)
                - ``confidence``    : float  (vote fraction in [0, 1])
        """
        self._sample_count += 1
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat()
        raw = float(aggregate_power)

        # --- step 1: input smoothing ---
        self._input_buffer.append(raw)
        smoothed = _smooth(self._input_buffer, self._smoothing_method)

        # --- step 2: combinatorial search with transition penalty ---
        diffs = np.abs(smoothed - self._combo_powers)  # (n_combos,)

        if self._committed_combo_idx is not None:
            # Apply penalty to every combination except the currently committed one.
            penalty_mask = np.ones(self.n_combinations, dtype=float)
            penalty_mask[self._committed_combo_idx] = 0.0
            penalised = diffs + penalty_mask * self._transition_penalty
        else:
            penalised = diffs

        best_idx = int(np.argmin(penalised))

        # --- step 3: hysteresis ---
        if self._committed_combo_idx is None:
            self._committed_combo_idx = best_idx
        else:
            committed_cost = float(diffs[self._committed_combo_idx])
            best_cost = float(diffs[best_idx])
            # Switch only if the improvement exceeds the dead zone.
            if committed_cost - best_cost > self._hysteresis_margin:
                self._committed_combo_idx = best_idx

        committed_combo = self._combinations[self._committed_combo_idx]

        # --- step 4: update per-appliance state buffers ---
        for j, (name, _hmm, _means, _labels, _idx_to_label) in enumerate(self._app_info):
            self._state_buffers[name].append(committed_combo[j])

        # --- step 5: majority vote and build output ---
        per_appliance: dict[str, dict] = {}
        for j, (name, _hmm, means, _labels, idx_to_label) in enumerate(self._app_info):
            buf = list(self._state_buffers[name])
            n_states = len(means)
            counts = np.bincount(buf, minlength=n_states)
            stable_state = int(np.argmax(counts))
            confidence = float(counts[stable_state] / len(buf))
            per_appliance[name] = {
                "state_label": idx_to_label.get(stable_state, str(stable_state)),
                "state_index": stable_state,
                "estimated_power": float(means[stable_state]),
                "confidence": round(confidence, 4),
            }

        return {
            "timestamp": ts,
            "raw_power": raw,
            "smoothed_power": round(smoothed, 3),
            "warmed_up": self.is_warmed_up,
            "appliances": per_appliance,
        }

    def reset(self) -> None:
        """Clear all internal buffers and reset inference state.

        Useful when the incoming power stream is interrupted or restarted.
        """
        self._input_buffer.clear()
        for buf in self._state_buffers.values():
            buf.clear()
        self._committed_combo_idx = None
        self._sample_count = 0
        logger.debug("RealTimeNILM buffers reset.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RealTimeNILM(house={self._house_number}, "
            f"appliances={self.appliances}, "
            f"combinations={self.n_combinations}, "
            f"input_buf={self._input_buffer_size}, "
            f"state_buf={self._state_buffer_size})"
        )
