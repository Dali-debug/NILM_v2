"""
utils.py
--------
Shared utilities for the REFIT NILM pipeline.

Consolidates helpers that were previously duplicated across modules:
- ``state_labels()``   replaces ``_state_labels()`` in train_hmm.py
                       and ``_state_names()`` in plot_appliance_signatures.py
- ``get_logger()``     provides a consistently formatted logger for any module
- ``ensure_dir()``     creates a directory and returns its path
"""

from __future__ import annotations

import logging
import os


# ---------------------------------------------------------------------------
# State naming
# ---------------------------------------------------------------------------

def state_labels(n_states: int) -> list[str]:
    """Return human-readable state labels for *n_states* states.

    Examples
    --------
    >>> state_labels(2)
    ['OFF', 'ON']
    >>> state_labels(3)
    ['OFF', 'LOW', 'HIGH']
    >>> state_labels(4)
    ['state_0', 'state_1', 'state_2', 'state_3']
    """
    if n_states == 2:
        return ["OFF", "ON"]
    if n_states == 3:
        return ["OFF", "LOW", "HIGH"]
    return [f"state_{i}" for i in range(n_states)]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger for *name*.

    The logger is idempotent: calling it multiple times with the same *name*
    returns the same logger without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int
        Initial log level (``logging.DEBUG``, ``logging.INFO``, …).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_log_level(level: int) -> None:
    """Set the log level on all NILM pipeline loggers at once.

    Call this from ``run_nilm.py`` when ``--verbose`` is supplied::

        import logging
        from utils import set_log_level
        set_log_level(logging.DEBUG)
    """
    for name in (
        "preprocessing",
        "train_hmm",
        "disaggregate",
        "pipeline.realtime",
        "refit_metadata",
        "plot_utils",
        "plot_appliance_signatures",
        "plot_prf_metrics",
    ):
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create *path* (and all parents) if it does not exist.

    Returns *path* so callers can do::

        out = os.path.join(ensure_dir(plots_dir), "figure.png")
    """
    os.makedirs(path, exist_ok=True)
    return path
