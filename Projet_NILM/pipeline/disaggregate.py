"""
disaggregate.py
---------------
Estimate the on/off state and power level of individual appliances from
REFIT sub-metering data (or from the aggregate signal in NILM mode).

Two modes are supported:

1. **Sub-metering mode** (default, recommended for evaluation):
   Each appliance's own power column is decoded through its trained HMM.

2. **NILM mode** (``--nilm`` flag):
   Only the aggregate power column is used.  A combinatorial search finds
   the per-appliance state combination whose predicted sum best matches the
   observed aggregate at each time step.

Usage
-----
    python disaggregate.py --house ../Processed_Data_CSV/House_3.csv
    python disaggregate.py --house ../Processed_Data_CSV/House_3.csv --nilm
    python disaggregate.py --house ../Processed_Data_CSV/House_3.csv \\
                           --appliances kettle microwave fridge tv \\
                           --limit 5000 --plot
"""

from __future__ import annotations

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MAX_NILM_COMBINATIONS, POWER_ON_THRESHOLD
from visualization.plot_utils import save_fig, style_axes
from .preprocessing import preprocess_house
from data.refit_metadata import get_appliance_column, parse_house_number
from .train_hmm import load_models, reconstruct_hmm
from utils import get_logger, state_labels

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_states(model, series: pd.Series) -> np.ndarray:
    """Run Viterbi on *series* using *model* and return the state sequence."""
    X = series.values.reshape(-1, 1)
    return model.predict(X)


def _state_to_label(state_idx: int, labels: list[str]) -> str:
    if 0 <= state_idx < len(labels):
        return labels[state_idx]
    return str(state_idx)


def _semantic_state_label_map(model, labels: list[str]) -> dict[int, str]:
    """Map raw HMM state index to semantic labels ordered by mean power.

    HMM state ids are permutation-invariant; this remaps them so the smallest
    mean maps to OFF, then LOW, then HIGH (when available).
    """
    n_states = model.n_components
    base_labels = list(labels) if len(labels) == n_states else state_labels(n_states)
    means = model.means_.flatten().astype(float)
    order = np.argsort(means)

    idx_to_label: dict[int, str] = {}
    for rank, state_idx in enumerate(order):
        label = base_labels[rank] if rank < len(base_labels) else f"state_{rank}"
        idx_to_label[int(state_idx)] = label
    return idx_to_label


def _states_to_semantic_labels(state_seq: np.ndarray,
                                idx_to_label: dict[int, str]) -> list[str]:
    """Convert a raw state sequence to semantic labels using idx_to_label."""
    return [idx_to_label.get(int(s), str(int(s))) for s in state_seq]


# ---------------------------------------------------------------------------
# Sub-metering disaggregation
# ---------------------------------------------------------------------------

def disaggregate_submetering(df: pd.DataFrame,
                              models: dict,
                              house_number: int,
                              target_appliances: list[str]) -> pd.DataFrame:
    """Decode hidden states using each appliance's own power column.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed REFIT DataFrame (all columns available).
    models : dict
        Mapping ``appliance → dict`` (from ``load_models``).
    house_number : int
    target_appliances : list of str

    Returns
    -------
    pd.DataFrame
        Columns: ``<appliance>_power``, ``<appliance>_state``,
        ``<appliance>_state_label`` for each appliance.
    """
    results = pd.DataFrame(index=df.index)

    for appliance in target_appliances:
        if appliance not in models:
            logger.warning("No model for '%s' - skipping disaggregation.", appliance)
            continue

        col = get_appliance_column(house_number, appliance)
        if col is None or col not in df.columns:
            logger.warning(
                "Column for '%s' not in DataFrame (house %d) - skipping.",
                appliance, house_number,
            )
            continue

        model_data = models[appliance]
        hmm = reconstruct_hmm(model_data)
        labels = model_data.get("state_labels", state_labels(hmm.n_components))
        idx_to_label = _semantic_state_label_map(hmm, labels)

        logger.info("Decoding states for '%s' (col=%s) ...", appliance, col)
        series = df[col]
        state_seq = _decode_states(hmm, series)
        state_labels_seq = _states_to_semantic_labels(state_seq, idx_to_label)

        results[f"{appliance}_power"] = series.values
        results[f"{appliance}_state"] = state_seq
        results[f"{appliance}_state_label"] = state_labels_seq
        logger.info("  '%s' OK | unique states: %s", appliance, sorted(set(state_labels_seq)))

    return results


# ---------------------------------------------------------------------------
# NILM disaggregation (aggregate-only)
# ---------------------------------------------------------------------------

def disaggregate_nilm(df: pd.DataFrame,
                      models: dict,
                      house_number: int,
                      target_appliances: list[str]) -> pd.DataFrame:
    """Estimate appliance states from the aggregate power signal only (NILM).

    For each time step the combination of per-appliance states whose
    predicted total power is closest to the observed aggregate is chosen.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed REFIT DataFrame.
    models : dict
    house_number : int
    target_appliances : list of str

    Returns
    -------
    pd.DataFrame
    """
    app_info = []
    for appliance in target_appliances:
        if appliance not in models:
            continue
        model_data = models[appliance]
        hmm = reconstruct_hmm(model_data)
        labels = model_data.get("state_labels", state_labels(hmm.n_components))
        means = hmm.means_.flatten().tolist()
        idx_to_label = _semantic_state_label_map(hmm, labels)
        app_info.append((appliance, hmm, means, labels, idx_to_label))

    if not app_info:
        logger.warning("No models available for NILM - skipping.")
        return pd.DataFrame(index=df.index)

    aggregate = df["Aggregate"].values
    state_ranges = [range(len(info[2])) for info in app_info]
    combinations = list(itertools.product(*state_ranges))

    n_combinations = len(combinations)
    if n_combinations > MAX_NILM_COMBINATIONS:
        logger.warning(
            "%d state combinations - NILM may be slow or run out of memory. "
            "Consider reducing the number of appliances or states.",
            n_combinations,
        )

    # Vectorised computation: predicted aggregate power for every combination
    means_matrix = np.array([
        [app_info[j][2][combo[j]] for combo in combinations]
        for j in range(len(app_info))
    ])
    combo_powers = means_matrix.sum(axis=0)  # (n_combos,)

    # aggregate: (n_samples,)  combo_powers: (n_combos,)
    diffs = np.abs(aggregate[:, None] - combo_powers[None, :])  # (n, n_combos)
    best_combo_idx = np.argmin(diffs, axis=1)                    # (n,)

    results = pd.DataFrame(index=df.index)
    results["aggregate_power"] = aggregate

    for j, (appliance, hmm, means, labels, idx_to_label) in enumerate(app_info):
        state_seq = np.array([combinations[ci][j] for ci in best_combo_idx])
        state_labels_seq = _states_to_semantic_labels(state_seq, idx_to_label)
        estimated_power = np.array([means[s] for s in state_seq])

        results[f"{appliance}_power_est"] = estimated_power
        results[f"{appliance}_state"] = state_seq
        results[f"{appliance}_state_label"] = state_labels_seq
        logger.info("NILM '%s': unique states: %s", appliance, sorted(set(state_labels_seq)))

    return results


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_state_events(results: pd.DataFrame,
                        target_appliances: list[str]) -> dict[str, pd.DataFrame]:
    """Detect state transition events for each appliance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are appliance names; values are DataFrames with columns
        ``["time", "from_state", "to_state", "power_w"]``.
    """
    events_by_appliance: dict[str, pd.DataFrame] = {}

    for appliance in target_appliances:
        label_col = f"{appliance}_state_label"
        if label_col not in results.columns:
            continue

        power_col = (f"{appliance}_power"
                     if f"{appliance}_power" in results.columns
                     else f"{appliance}_power_est")

        labels = results[label_col]
        prev_labels = labels.shift(1)
        transitions = labels.ne(prev_labels)
        if len(transitions) > 0:
            transitions.iloc[0] = False

        event_times = results.index[transitions.fillna(False)]
        if len(event_times) == 0:
            events_by_appliance[appliance] = pd.DataFrame(
                columns=["time", "from_state", "to_state", "power_w"]
            )
            continue

        if power_col in results.columns:
            power_values = results.loc[event_times, power_col].astype(float).values
        else:
            power_values = np.full(len(event_times), np.nan, dtype=float)

        events_by_appliance[appliance] = pd.DataFrame({
            "time": event_times,
            "from_state": prev_labels.loc[event_times].values,
            "to_state": labels.loc[event_times].values,
            "power_w": power_values,
        })

    return events_by_appliance


def print_event_summary(events_by_appliance: dict[str, pd.DataFrame],
                        events_per_appliance: int = 5) -> None:
    """Print more than one event per appliance for quick inspection."""
    print("\n--- Event Detection Summary ---")
    for appliance, events_df in events_by_appliance.items():
        if events_df.empty:
            print(f"  {appliance:15s}: no detected state transitions")
            continue

        print(f"  {appliance:15s}: {len(events_df):,} transitions detected")
        for _, row in events_df.head(events_per_appliance).iterrows():
            print(
                f"    {row['time']} | {row['from_state']} -> {row['to_state']} "
                f"| {row['power_w']:.1f} W"
            )


def plot_event_windows(results: pd.DataFrame,
                       events_by_appliance: dict[str, pd.DataFrame],
                       house_number: int,
                       plots_dir: str | None = None,
                       events_per_appliance: int = 5,
                       event_window: int = 120) -> None:
    """Save event-centric plots with multiple events per appliance."""
    if plots_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")

    for appliance, events_df in events_by_appliance.items():
        if events_df.empty:
            continue

        power_col = (f"{appliance}_power"
                     if f"{appliance}_power" in results.columns
                     else f"{appliance}_power_est")
        if power_col not in results.columns:
            continue

        n_events = min(events_per_appliance, len(events_df))
        fig, axes = plt.subplots(n_events, 1, figsize=(12, 3 * n_events),
                                 sharex=False)
        if n_events == 1:
            axes = [axes]

        for i in range(n_events):
            event_row = events_df.iloc[i]
            t_event = event_row["time"]
            event_pos = results.index.get_loc(t_event)
            start = max(0, event_pos - event_window)
            end = min(len(results), event_pos + event_window + 1)

            segment = results.iloc[start:end]
            x = np.arange(start, end)

            axes[i].plot(x, segment[power_col].values,
                         color="steelblue", linewidth=0.9)
            axes[i].axvline(event_pos, color="crimson", linestyle="--",
                            linewidth=1.2)
            style_axes(axes[i],
                       title=f"Event {i+1}: {event_row['from_state']} -> "
                             f"{event_row['to_state']} at {t_event}",
                       ylabel="Power (W)")

        axes[-1].set_xlabel("Sample index")
        plt.tight_layout()
        out_path = os.path.join(plots_dir,
                                f"house{house_number}_{appliance}_events.png")
        save_fig(fig, out_path, dpi=110)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: pd.DataFrame,
                 target_appliances: list[str],
                 house_number: int,
                 plots_dir: str | None = None,
                 limit: int = 2000) -> None:
    """Save time-series plots for each appliance's power and decoded states.

    Parameters
    ----------
    results : pd.DataFrame
        Output from ``disaggregate_submetering`` or ``disaggregate_nilm``.
    target_appliances : list of str
    house_number : int
    plots_dir : str, optional
    limit : int
        Maximum number of samples to plot.
    """
    if plots_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")

    plot_df = results.iloc[:limit]
    time_axis = range(len(plot_df))

    for appliance in target_appliances:
        power_col = (f"{appliance}_power"
                     if f"{appliance}_power" in results.columns
                     else f"{appliance}_power_est")
        label_col = f"{appliance}_state_label"

        if power_col not in results.columns:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(time_axis, plot_df[power_col].values,
                     color="steelblue", linewidth=0.8, label="Power (W)")
        axes[0].legend(loc="upper right")
        style_axes(axes[0],
                   title=f"House {house_number} — {appliance.capitalize()}: Power consumption",
                   ylabel="Power (W)")

        if label_col in plot_df.columns:
            unique_labels = sorted(set(plot_df[label_col]))
            label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
            state_int = plot_df[label_col].map(label_to_int).values

            axes[1].step(time_axis, state_int, where="post",
                         color="darkorange", linewidth=1.2)
            axes[1].set_yticks(range(len(unique_labels)))
            axes[1].set_yticklabels(unique_labels)
            style_axes(axes[1],
                       title=f"House {house_number} — {appliance.capitalize()}: Decoded state",
                       xlabel="Time (samples)",
                       ylabel="State")

        plt.tight_layout()
        out_path = os.path.join(plots_dir,
                                f"house{house_number}_{appliance}_states.png")
        save_fig(fig, out_path, dpi=100)


# ---------------------------------------------------------------------------
# Evaluation (when ground truth is available)
# ---------------------------------------------------------------------------

def evaluate_results(results: pd.DataFrame,
                     df: pd.DataFrame,
                     models: dict,
                     house_number: int,
                     target_appliances: list[str]) -> None:
    """Print a simple accuracy summary comparing decoded states with ground truth.

    Ground-truth states are obtained by thresholding the sub-meter power:
    a sample is ON if its power exceeds ``config.POWER_ON_THRESHOLD`` Watts.
    """
    print("\n--- Evaluation Summary ---")
    for appliance in target_appliances:
        state_col = f"{appliance}_state_label"
        if state_col not in results.columns:
            continue

        col = get_appliance_column(house_number, appliance)
        if col is None or col not in df.columns:
            continue

        # Ground truth: ON if power > POWER_ON_THRESHOLD
        gt_on = (df[col] > POWER_ON_THRESHOLD).astype(int)
        pred_on = (results[state_col] != "OFF").astype(int)

        common_idx = results.index.intersection(df.index)
        gt_on = gt_on.loc[common_idx]
        pred_on = pred_on.loc[common_idx]

        accuracy = (gt_on.values == pred_on.values).mean()
        print(f"  {appliance:15s}: ON/OFF accuracy = {accuracy:.3f} "
              f"({len(common_idx):,} samples)")


# ---------------------------------------------------------------------------
# Main disaggregation pipeline
# ---------------------------------------------------------------------------

def run_disaggregation(house_csv: str,
                       target_appliances: list[str] | None = None,
                       nilm_mode: bool = False,
                       limit: int | None = None,
                       plot: bool = True,
                       models_dir: str | None = None,
                       plots_dir: str | None = None,
                       train_house_number: int | None = None,
                       plot_preprocessing: bool = False,
                       preprocessing_plot_limit: int = 3000,
                       detect_events: bool = False,
                       events_per_appliance: int = 5,
                       event_window: int = 120) -> pd.DataFrame:
    """Full disaggregation pipeline.

    Parameters
    ----------
    house_csv : str
        Path to REFIT house CSV (test/evaluation house).
    target_appliances : list of str, optional
        Defaults to ``config.DEFAULT_APPLIANCES``.
    nilm_mode : bool
        If True use only the aggregate column (true NILM).
    limit : int, optional
        Truncate data to this many samples.
    plot : bool
        Generate and save plots.
    models_dir : str, optional
        Directory containing the saved model JSON files.
    plots_dir : str, optional
        Directory to write plot PNG files.
    train_house_number : int, optional
        House number the models were trained on (for cross-house evaluation).

    Returns
    -------
    pd.DataFrame
        State and power estimates for each appliance.
    """
    from config import DEFAULT_APPLIANCES
    if target_appliances is None:
        target_appliances = list(DEFAULT_APPLIANCES)

    test_house_number = parse_house_number(house_csv)
    if train_house_number is None:
        train_house_number = test_house_number

    cross_house = train_house_number != test_house_number
    if cross_house:
        logger.info(
            "=== Disaggregating House %d (models from House %d) ===",
            test_house_number, train_house_number,
        )
    else:
        logger.info("=== Disaggregating House %d ===", test_house_number)

    # 1. Load and preprocess
    test_columns = ["Aggregate"]
    for appliance in target_appliances:
        col = get_appliance_column(test_house_number, appliance)
        if col is not None:
            test_columns.append(col)
    test_columns = sorted(set(test_columns))

    df = preprocess_house(
        house_csv,
        max_rows=limit,
        plot_preprocessing=plot_preprocessing,
        preprocessing_plot_columns=test_columns,
        preprocessing_plots_dir=plots_dir,
        preprocessing_plot_limit=preprocessing_plot_limit,
        preprocessing_plot_tag="test",
    )
    if limit:
        df = df.iloc[:limit]
        logger.info("Data limited to first %d samples.", limit)

    # 2. Load models (from train house directory)
    logger.info("Loading models ...")
    models = load_models(train_house_number, target_appliances,
                         models_dir=models_dir)
    if not models:
        raise RuntimeError(
            f"No trained models found for House {train_house_number}. "
            "Run train_hmm.py first."
        )

    # 3. Disaggregate
    logger.info("Decoding states ...")
    if nilm_mode:
        results = disaggregate_nilm(df, models, test_house_number, target_appliances)
    else:
        results = disaggregate_submetering(df, models, test_house_number,
                                           target_appliances)

    # 4. Evaluate (only meaningful in sub-metering mode)
    if not nilm_mode:
        evaluate_results(results, df, models, test_house_number, target_appliances)

    # 5. Plot
    if plot:
        logger.info("Generating plots ...")
        plot_results(results, target_appliances, test_house_number,
                     plots_dir=plots_dir)

    # 6. Event detection
    if detect_events:
        events_by_appliance = detect_state_events(results, target_appliances)
        print_event_summary(events_by_appliance,
                            events_per_appliance=events_per_appliance)
        plot_event_windows(
            results=results,
            events_by_appliance=events_by_appliance,
            house_number=test_house_number,
            plots_dir=plots_dir,
            events_per_appliance=events_per_appliance,
            event_window=event_window,
        )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disaggregate REFIT appliances using trained HMMs."
    )
    parser.add_argument("--house", required=True,
                        help="Path to the REFIT house CSV file.")
    parser.add_argument("--appliances", nargs="+",
                        default=["kettle", "microwave", "fridge", "tv"])
    parser.add_argument("--nilm", action="store_true",
                        help="Use aggregate-only NILM mode.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-preprocessing", action="store_true")
    parser.add_argument("--preprocessing-plot-limit", type=int, default=3000)
    parser.add_argument("--detect-events", action="store_true")
    parser.add_argument("--events-per-appliance", type=int, default=5)
    parser.add_argument("--event-window", type=int, default=120)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    if args.verbose:
        import logging
        from utils import set_log_level
        set_log_level(logging.DEBUG)

    results = run_disaggregation(
        house_csv=args.house,
        target_appliances=args.appliances,
        nilm_mode=args.nilm,
        limit=args.limit,
        plot=not args.no_plot,
        plot_preprocessing=args.plot_preprocessing,
        preprocessing_plot_limit=args.preprocessing_plot_limit,
        detect_events=args.detect_events,
        events_per_appliance=args.events_per_appliance,
        event_window=args.event_window,
    )

    print("\n--- Sample Output (first 5 rows) ---")
    print(results.head().to_string())
