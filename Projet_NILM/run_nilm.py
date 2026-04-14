"""
run_nilm.py
-----------
Main entry point for the REFIT NILM pipeline.

This script orchestrates the full pipeline:
  1. Load and preprocess a REFIT house CSV.
  2. Train per-appliance Gaussian HMMs (saved to ``models/``).
  3. Decode the state of each appliance at every time step.
  4. Generate and save plots to ``plots/``.
  5. Print a state-summary table.

Usage
-----
    # Full pipeline on a single house (train + disaggregate on same house):
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv

    # Cross-house: train on House 9, test/disaggregate on House 3:
    python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv \\
                       --test-house  ../Processed_Data_CSV/House_3.csv

    # Training only on House 9:
    python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv --mode train

    # Disaggregation only on House 3 using models trained on House 9:
    python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv \\
                       --test-house  ../Processed_Data_CSV/House_3.csv \\
                       --mode disaggregate

    # True NILM mode (use aggregate signal only, no sub-metering):
    python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv \\
                       --test-house  ../Processed_Data_CSV/House_3.csv --nilm

    # Limit data for a quick test:
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --limit 5000

    # Custom appliance selection:
    python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv \\
                       --test-house  ../Processed_Data_CSV/House_3.csv \\
                       --appliances kettle microwave fridge tv

    # Verbose / debug output:
    python run_nilm.py --house ../Processed_Data_CSV/House_3.csv --verbose

Appliance column mapping (verified ground truth)
-------------------------------------------------
House 3:  Appliance1=Toaster, Appliance2=Fridge-Freezer, Appliance3=Freezer,
          Appliance4=Tumble Dryer, Appliance5=Dishwasher, Appliance6=Washing Machine,
          Appliance7=Television, Appliance8=Microwave, Appliance9=Kettle

House 9:  Appliance1=Fridge-Freezer, Appliance2=Washer Dryer, Appliance3=Washing Machine,
          Appliance4=Dishwasher, Appliance5=Television Site, Appliance6=Microwave,
          Appliance7=Kettle, Appliance8=Hi-Fi, Appliance9=Electric Heater
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from config import DEFAULT_APPLIANCES, MAX_TRAIN_SAMPLES, PREPROCESSING_PLOT_LIMIT
from data.refit_metadata import HOUSE_APPLIANCES, get_appliance_column, parse_house_number
from pipeline.train_hmm import run_training
from pipeline.disaggregate import run_disaggregation
from utils import get_logger, set_log_level

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State-summary printers
# ---------------------------------------------------------------------------

def print_state_summary(results: pd.DataFrame, target_appliances: list[str],
                        house_number: int) -> None:
    """Print a concise state-occupancy table for every appliance."""
    print("\n" + "=" * 60)
    print(f"  STATE SUMMARY - House {house_number}")
    print("=" * 60)

    rows = []
    for appliance in target_appliances:
        label_col = f"{appliance}_state_label"
        if label_col not in results.columns:
            continue
        counts = results[label_col].value_counts()
        total = len(results)
        row = {"Appliance": appliance.capitalize()}
        for state, cnt in counts.items():
            row[state] = f"{cnt:>8,}  ({100 * cnt / total:5.1f}%)"
        rows.append(row)

    if not rows:
        print("  No results to display.")
        return

    summary_df = pd.DataFrame(rows).set_index("Appliance").fillna("-")
    print(summary_df.to_string())
    print("=" * 60)


def print_recent_states(results: pd.DataFrame, target_appliances: list[str],
                        n_rows: int = 10) -> None:
    """Print the last *n_rows* state labels for each appliance."""
    label_cols = [f"{a}_state_label" for a in target_appliances
                  if f"{a}_state_label" in results.columns]
    if not label_cols:
        return

    print(f"\n--- Last {n_rows} samples ---")
    display = results[label_cols].tail(n_rows).copy()
    display.columns = [c.replace("_state_label", "").capitalize()
                       for c in display.columns]
    print(display.to_string())


def print_appliance_map(house_number: int, target_appliances: list[str]) -> None:
    """Print the column→device mapping for *house_number*."""
    if house_number not in HOUSE_APPLIANCES:
        return
    house_map = {
        f"Appliance{i+1}": name
        for i, name in enumerate(HOUSE_APPLIANCES[house_number])
    }
    print(f"\nHouse {house_number} appliance map:")
    for col, name in house_map.items():
        print(f"  {col}: {name}")

    print(f"\nTarget appliance -> column mapping for House {house_number}:")
    for appliance in target_appliances:
        col = get_appliance_column(house_number, appliance)
        if col is not None:
            device_label = house_map.get(col, "?")
            print(f"  {appliance:20s} -> {col} ({device_label})")
        else:
            print(f"  {appliance:20s} -> [NOT FOUND in House {house_number}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "REFIT NILM Pipeline — Disaggregate household appliances "
            "using Hidden Markov Models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--house",
        help=(
            "Path to a REFIT house CSV used for both training and testing. "
            "Kept for backward compatibility; prefer --train-house / --test-house."
        ),
    )
    parser.add_argument(
        "--train-house", metavar="CSV",
        help="Path to the REFIT house CSV used for training HMMs.",
    )
    parser.add_argument(
        "--test-house", metavar="CSV",
        help="Path to the REFIT house CSV used for disaggregation/evaluation.",
    )
    parser.add_argument(
        "--appliances", nargs="+", default=list(DEFAULT_APPLIANCES),
        metavar="APPLIANCE",
        help=f"Canonical appliance names to process. Default: {DEFAULT_APPLIANCES}.",
    )
    parser.add_argument(
        "--mode", choices=["all", "train", "disaggregate"], default="all",
        help=(
            "'all' = train then disaggregate (default); "
            "'train' = training only; "
            "'disaggregate' = disaggregation only (requires saved models)."
        ),
    )
    parser.add_argument(
        "--nilm", action="store_true",
        help="Run in true NILM mode: use aggregate signal only.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N samples (useful for quick tests).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--n-states", type=int, default=None,
        help="Override number of HMM hidden states for all appliances.",
    )
    parser.add_argument(
        "--fridge-states", type=int, default=None,
        help="Override number of HMM hidden states specifically for fridge.",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=MAX_TRAIN_SAMPLES,
        help=f"Maximum training samples per appliance (default: {MAX_TRAIN_SAMPLES}).",
    )
    parser.add_argument(
        "--plot-preprocessing", action="store_true",
        help="Generate raw vs preprocessed plots for train/test signals.",
    )
    parser.add_argument(
        "--preprocessing-plot-limit", type=int, default=PREPROCESSING_PLOT_LIMIT,
        help=f"Maximum samples in preprocessing plots (default: {PREPROCESSING_PLOT_LIMIT}).",
    )
    parser.add_argument(
        "--detect-events", action="store_true",
        help="Detect state transitions and print/plot events per appliance.",
    )
    parser.add_argument(
        "--events-per-appliance", type=int, default=5,
        help="Number of events to display and plot for each appliance.",
    )
    parser.add_argument(
        "--event-window", type=int, default=120,
        help="Half-window size (in samples) around each detected event plot.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging output.",
    )
    return parser


def main() -> pd.DataFrame | None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        set_log_level(logging.DEBUG)

    # Resolve train/test CSV paths
    train_csv = args.train_house or args.house
    test_csv = args.test_house or args.train_house or args.house

    if train_csv is None:
        parser.error(
            "Provide at least one of --house, --train-house, or --test-house."
        )

    for label, path in [("train", train_csv), ("test", test_csv)]:
        if not os.path.isfile(path):
            logger.error("%s file not found: %r", label.capitalize(), path)
            print(
                "\nPlease download the REFIT dataset and place CSV files in "
                "Processed_Data_CSV/ — see the project README for instructions."
            )
            sys.exit(1)

    train_house_number = parse_house_number(train_csv)
    test_house_number = parse_house_number(test_csv)
    appliances = args.appliances

    appliance_n_states: dict = {}
    if args.fridge_states is not None:
        appliance_n_states["fridge"] = args.fridge_states

    cross_house = train_house_number != test_house_number

    # Print setup header
    print("\n" + "=" * 60)
    print("  NILM Pipeline Setup")
    print("=" * 60)
    print(f"  Train house : {train_house_number}  ({train_csv})")
    print(f"  Test house  : {test_house_number}  ({test_csv})")
    print(f"  Mode        : {args.mode}")
    print(f"  Appliances  : {appliances}")
    if cross_house:
        print("  Cross-house : YES - models trained on House "
              f"{train_house_number}, evaluated on House {test_house_number}")
    print("=" * 60)

    print_appliance_map(train_house_number, appliances)
    if cross_house:
        print_appliance_map(test_house_number, appliances)

    # ---- Training ----
    if args.mode in ("all", "train"):
        run_training(
            house_csv=train_csv,
            target_appliances=appliances,
            n_states_override=args.n_states,
            sample_limit=args.sample_limit,
            preprocess_max_rows=args.limit,
            appliance_n_states=appliance_n_states or None,
            plot_preprocessing=args.plot_preprocessing,
            preprocessing_plot_limit=args.preprocessing_plot_limit,
            preprocessing_plot_tag="train",
        )

    # ---- Disaggregation ----
    if args.mode in ("all", "disaggregate"):
        results = run_disaggregation(
            house_csv=test_csv,
            target_appliances=appliances,
            nilm_mode=args.nilm,
            limit=args.limit,
            plot=not args.no_plot,
            train_house_number=train_house_number,
            plot_preprocessing=args.plot_preprocessing,
            preprocessing_plot_limit=args.preprocessing_plot_limit,
            detect_events=args.detect_events,
            events_per_appliance=args.events_per_appliance,
            event_window=args.event_window,
        )

        print_state_summary(results, appliances, test_house_number)
        print_recent_states(results, appliances)

        print("\n[DONE] Plots saved to: plots/")
        print("       Models saved to: models/")
        return results

    return None


if __name__ == "__main__":
    main()
