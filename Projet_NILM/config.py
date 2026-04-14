"""
config.py
---------
Central configuration for the REFIT NILM pipeline.

All magic numbers and tunable defaults live here so every module reads
from a single source of truth.  Import what you need::

    from config import POWER_ON_THRESHOLD, DEFAULT_APPLIANCES
"""

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
RESAMPLE_RATE: str = "8s"           # REFIT native sampling interval
HAMPEL_WINDOW: int = 15             # Half-width of Hampel filter window
HAMPEL_THRESHOLD: float = 3.0       # n-sigma threshold for Hampel identifier
MAX_INTERPOLATION_GAP: int = 10     # Max consecutive NaN time steps to fill
PREPROCESSING_PLOT_LIMIT: int = 3000  # Max samples in raw-vs-preprocessed plots

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MAX_TRAIN_SAMPLES: int = 50_000     # Sub-sample cap for HMM EM training
HMM_N_ITER: int = 100               # Maximum Baum-Welch iterations
HMM_RANDOM_STATE: int = 42          # Reproducibility seed

# Default number of hidden states per appliance type
DEFAULT_N_STATES: dict = {
    "kettle":    2,   # OFF / ON  (short burst, high power)
    "microwave": 2,   # OFF / ON
    "fridge":    3,   # OFF / LOW / HIGH (compressor cycling)
    "tv":        2,   # OFF / ON
}

# ---------------------------------------------------------------------------
# Disaggregation / evaluation
# ---------------------------------------------------------------------------
# Watts — appliance considered ON if its sub-meter power exceeds this value.
# Used for ground-truth binarisation in evaluate_results() and plot_prf_metrics.py.
POWER_ON_THRESHOLD: float = 10.0

# Warn when the NILM state-combination space exceeds this size.
MAX_NILM_COMBINATIONS: int = 5_000

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------
DEFAULT_APPLIANCES: list = ["kettle", "microwave", "fridge", "tv"]

# ---------------------------------------------------------------------------
# Real-time inference
# ---------------------------------------------------------------------------
# Rolling window over recent aggregate samples used for input smoothing.
# Larger values increase latency but reduce the effect of transient spikes.
RT_INPUT_BUFFER_SIZE: int = 15

# Per-appliance deque length for majority-vote state stabilization.
# Larger values produce smoother outputs but slow response to real transitions.
RT_STATE_BUFFER_SIZE: int = 9

# Smoothing method applied to the input buffer.
# "median"       — robust to outliers, preferred for noisy meters.
# "trimmed_mean" — trims the outer 25 % before averaging.
RT_SMOOTHING_METHOD: str = "median"

# Minimum improvement (Watts) that a new state combination must achieve over
# the currently committed combination before a state change is committed.
# Prevents oscillation near decision boundaries (hysteresis dead zone).
RT_HYSTERESIS_MARGIN: float = 30.0

# Extra cost (Watts) added to every state combination that differs from the
# currently committed one during the combinatorial search.  Creates temporal
# continuity so the inferred combination does not jump between consecutive
# samples when two combinations are nearly equally likely.
RT_TRANSITION_PENALTY: float = 50.0
