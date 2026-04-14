"""
train_hmm.py
------------
Train a Gaussian HMM for each target appliance using REFIT sub-metering data.

The trained models are saved as JSON files in the ``models/`` sub-directory
(relative to this script's location).

Usage (standalone)
------------------
    python train_hmm.py --house ../Processed_Data_CSV/House_3.csv \\
                        [--appliances kettle microwave fridge tv] \\
                        [--n-states 2] [--sample-limit 50000]

The trained models will be saved to:
    models/<house_number>/<appliance_name>_hmm.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import (
    DEFAULT_N_STATES,
    HMM_N_ITER,
    HMM_RANDOM_STATE,
    MAX_TRAIN_SAMPLES,
)
from .preprocessing import preprocess_house
from data.refit_metadata import get_appliance_column, parse_house_number
from utils import get_logger, state_labels

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_appliance_hmm(power_series: pd.Series,
                        n_states: int,
                        n_iter: int = HMM_N_ITER,
                        sample_limit: int = MAX_TRAIN_SAMPLES,
                        random_state: int = HMM_RANDOM_STATE) -> GaussianHMM:
    """Fit a Gaussian HMM to the power consumption of one appliance.

    Parameters
    ----------
    power_series : pd.Series
        Watt readings for one appliance.
    n_states : int
        Number of hidden states (e.g. 2 for OFF/ON, 3 for OFF/LOW/HIGH).
    n_iter : int
        Maximum EM iterations.  Defaults to ``config.HMM_N_ITER``.
    sample_limit : int
        Maximum number of samples to use (sub-sampled randomly if exceeded).
        Defaults to ``config.MAX_TRAIN_SAMPLES``.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    GaussianHMM
        Fitted HMM model.
    """
    rng = np.random.default_rng(random_state)

    values = power_series.dropna().values.astype(np.float64)

    if len(values) == 0:
        raise ValueError("power_series contains no non-NaN values.")

    if len(values) > sample_limit:
        idx = rng.choice(len(values), size=sample_limit, replace=False)
        idx.sort()
        values = values[idx]
        logger.debug("Sub-sampled to %d training points.", sample_limit)

    X = values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )

    # Sensible initialisation based on quantiles accelerates EM convergence.
    init_means = [np.percentile(values, q) for q in np.linspace(0, 100, n_states)]
    model.means_prior = np.array([[m] for m in init_means])

    model.fit(X)

    if not model.monitor_.converged:
        logger.warning(
            "HMM did not converge in %d iterations. "
            "Consider increasing --n-iter or checking data quality.",
            n_iter,
        )
    else:
        logger.debug("HMM converged after %d iterations.", len(model.monitor_.history))

    return model


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------

def _hmm_to_dict(model: GaussianHMM, appliance: str, n_states: int) -> dict:
    """Serialise a fitted GaussianHMM to a JSON-compatible dict."""
    return {
        "appliance": appliance,
        "n_states": n_states,
        "state_labels": state_labels(n_states),
        "startprob": model.startprob_.tolist(),
        "transmat": model.transmat_.tolist(),
        "means": model.means_.tolist(),
        "covars": model.covars_.tolist(),
    }


def save_models(models_dict: dict,
                house_number: int,
                models_dir: str | None = None) -> None:
    """Save trained HMM models to JSON files.

    Parameters
    ----------
    models_dict : dict
        Mapping ``appliance_name → GaussianHMM`` or ``appliance_name → dict``.
    house_number : int
    models_dir : str, optional
        Directory to save files.  Defaults to
        ``<script_dir>/models/<house_number>/``.

    Raises
    ------
    OSError
        If the directory cannot be created or the file cannot be written.
    """
    if models_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "..", "models", str(house_number))

    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create models directory {models_dir!r}: {exc}") from exc

    for appliance, model_data in models_dict.items():
        filepath = os.path.join(models_dir, f"{appliance}_hmm.json")
        if isinstance(model_data, GaussianHMM):
            n_states = model_data.n_components
            data = _hmm_to_dict(model_data, appliance, n_states)
        else:
            data = model_data
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved model -> %s", filepath)
        except OSError as exc:
            raise OSError(f"Cannot write model file {filepath!r}: {exc}") from exc


def load_models(house_number: int,
                appliances: list[str],
                models_dir: str | None = None) -> dict:
    """Load previously saved HMM models from JSON.

    Parameters
    ----------
    house_number : int
    appliances : list of str
        Canonical appliance names to load.
    models_dir : str, optional
        Directory containing the JSON files.

    Returns
    -------
    dict
        Mapping ``appliance_name → dict`` (raw JSON data).
    """
    if models_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "..", "models", str(house_number))

    loaded = {}
    for appliance in appliances:
        filepath = os.path.join(models_dir, f"{appliance}_hmm.json")
        if not os.path.exists(filepath):
            logger.warning("Model not found: %s", filepath)
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                loaded[appliance] = json.load(f)
            logger.debug("Loaded model: %s", filepath)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load model %s: %s", filepath, exc)

    return loaded


def reconstruct_hmm(model_dict: dict) -> GaussianHMM:
    """Reconstruct a GaussianHMM object from a saved model dict.

    Parameters
    ----------
    model_dict : dict
        Dict as returned by ``load_models``.

    Returns
    -------
    GaussianHMM
        Ready-to-use fitted model (predict, score, etc.).
    """
    n_states = model_dict["n_states"]
    model = GaussianHMM(n_components=n_states, covariance_type="full")
    model.startprob_ = np.array(model_dict["startprob"])
    model.transmat_ = np.array(model_dict["transmat"])
    model.means_ = np.array(model_dict["means"])
    model.covars_ = np.array(model_dict["covars"])
    return model


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(house_csv: str,
                 target_appliances: list[str] | None = None,
                 n_states_override: int | None = None,
                 sample_limit: int = MAX_TRAIN_SAMPLES,
                 models_dir: str | None = None,
                 preprocess_max_rows: int | None = None,
                 appliance_n_states: dict[str, int] | None = None,
                 plot_preprocessing: bool = False,
                 preprocessing_plot_limit: int = 3000,
                 preprocessing_plots_dir: str | None = None,
                 preprocessing_plot_tag: str = "train") -> dict:
    """Full training pipeline: load REFIT data → train HMMs → save.

    Parameters
    ----------
    house_csv : str
        Path to the REFIT house CSV file.
    target_appliances : list of str, optional
        Canonical appliance names to train.  Defaults to
        ``config.DEFAULT_APPLIANCES``.
    n_states_override : int or None
        If set, use this number of states for all appliances instead of
        the per-appliance defaults in ``config.DEFAULT_N_STATES``.
    sample_limit : int
        Maximum training samples per appliance.
    preprocess_max_rows : int or None
        If provided, read at most this many CSV rows before preprocessing.
    models_dir : str or None
        Where to save the models (defaults to ``models/<house_number>/``).
    appliance_n_states : dict, optional
        Per-appliance state count overrides, e.g. ``{"fridge": 2}``.

    Returns
    -------
    dict
        Mapping ``appliance_name → GaussianHMM``.
    """
    from config import DEFAULT_APPLIANCES
    if target_appliances is None:
        target_appliances = list(DEFAULT_APPLIANCES)

    house_number = parse_house_number(house_csv)
    logger.info("=== Training HMMs for House %d ===", house_number)

    appliance_columns = ["Aggregate"]
    for appliance in target_appliances:
        col = get_appliance_column(house_number, appliance)
        if col is not None:
            appliance_columns.append(col)
    appliance_columns = sorted(set(appliance_columns))

    # 1. Load and preprocess
    df = preprocess_house(
        house_csv,
        max_rows=preprocess_max_rows,
        plot_preprocessing=plot_preprocessing,
        preprocessing_plot_columns=appliance_columns,
        preprocessing_plots_dir=preprocessing_plots_dir,
        preprocessing_plot_limit=preprocessing_plot_limit,
        preprocessing_plot_tag=preprocessing_plot_tag,
    )

    # 2. Train per appliance
    trained_models: dict = {}
    for appliance in target_appliances:
        col = get_appliance_column(house_number, appliance)
        if col is None:
            logger.warning("'%s' not found in House %d - skipping.", appliance, house_number)
            continue
        if col not in df.columns:
            logger.warning("Column '%s' missing from DataFrame - skipping.", col)
            continue

        if appliance_n_states and appliance in appliance_n_states:
            n_states = appliance_n_states[appliance]
        elif n_states_override:
            n_states = n_states_override
        else:
            n_states = DEFAULT_N_STATES.get(appliance, 2)

        logger.info("Training '%s' (column=%s, states=%d) ...", appliance, col, n_states)
        series = df[col]
        try:
            model = train_appliance_hmm(series, n_states=n_states,
                                        sample_limit=sample_limit)
            trained_models[appliance] = model
            means_sorted = sorted(model.means_.flatten().tolist())
            logger.info(
                "  '%s' OK | means ~= %s",
                appliance, [round(m, 1) for m in means_sorted],
            )
        except Exception as exc:
            logger.error("Failed to train '%s': %s", appliance, exc)

    # 3. Save models
    logger.info("Saving models ...")
    save_models(trained_models, house_number, models_dir=models_dir)

    return trained_models


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HMMs for REFIT appliances."
    )
    parser.add_argument("--house", required=True,
                        help="Path to the REFIT house CSV file.")
    parser.add_argument("--appliances", nargs="+",
                        default=["kettle", "microwave", "fridge", "tv"],
                        help="Appliances to train.")
    parser.add_argument("--n-states", type=int, default=None,
                        help="Override number of HMM states for all appliances.")
    parser.add_argument("--fridge-states", type=int, default=None,
                        help="Override number of states specifically for fridge.")
    parser.add_argument("--sample-limit", type=int, default=MAX_TRAIN_SAMPLES,
                        help=f"Max training samples per appliance (default: {MAX_TRAIN_SAMPLES}).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Read at most this many CSV rows for quick tests.")
    parser.add_argument("--plot-preprocessing", action="store_true",
                        help="Save raw vs preprocessed signal plots.")
    parser.add_argument("--preprocessing-plot-limit", type=int,
                        default=3000,
                        help="Max number of samples in preprocessing plots.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    if args.verbose:
        import logging
        from utils import set_log_level
        set_log_level(logging.DEBUG)

    appliance_n_states: dict = {}
    if args.fridge_states is not None:
        appliance_n_states["fridge"] = args.fridge_states

    run_training(
        house_csv=args.house,
        target_appliances=args.appliances,
        n_states_override=args.n_states,
        sample_limit=args.sample_limit,
        preprocess_max_rows=args.max_rows,
        appliance_n_states=appliance_n_states or None,
        plot_preprocessing=args.plot_preprocessing,
        preprocessing_plot_limit=args.preprocessing_plot_limit,
    )
