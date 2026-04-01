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

Appliance column mapping (verified ground truth)
-------------------------------------------------
House 3:  Appliance1=Toaster, Appliance2=Fridge-Freezer, Appliance3=Freezer,
          Appliance4=Tumble Dryer, Appliance5=Dishwasher, Appliance6=Washing Machine,
          Appliance7=Television, Appliance8=Microwave, Appliance9=Kettle

House 9:  Appliance1=Fridge-Freezer, Appliance2=Washer Dryer, Appliance3=Washing Machine,
          Appliance4=Dishwasher, Appliance5=Television Site, Appliance6=Microwave,
          Appliance7=Kettle, Appliance8=Hi-Fi, Appliance9=Electric Heater
"""

import argparse
import sys
import os
import pandas as pd

from refit_metadata import (
    parse_house_number, get_appliance_column, HOUSE_APPLIANCES
)
from train_hmm import run_training
from disaggregate import run_disaggregation

DEFAULT_APPLIANCES = ["kettle", "microwave", "fridge", "tv"]


# ---------------------------------------------------------------------------
# State-summary printer
# ---------------------------------------------------------------------------

def print_state_summary(results: pd.DataFrame, target_appliances: list,
                        house_number: int):
    """Print a concise state-occupancy table for every appliance."""
    print("\n" + "=" * 60)
    print(f"  STATE SUMMARY — House {house_number}")
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

    summary_df = pd.DataFrame(rows).set_index("Appliance")
    summary_df = summary_df.fillna("—")
    print(summary_df.to_string())
    print("=" * 60)


def print_recent_states(results: pd.DataFrame, target_appliances: list,
                        n_rows: int = 10):
    """Print the last *n_rows* state labels for each appliance."""
    label_cols = [f"{a}_state_label" for a in target_appliances
                  if f"{a}_state_label" in results.columns]
    if not label_cols:
        return

    print(f"\n--- Last {n_rows} samples ---")
    display = results[label_cols].tail(n_rows)
    display.columns = [c.replace("_state_label", "").capitalize()
                       for c in display.columns]
    print(display.to_string())


def print_appliance_map(house_number: int, target_appliances: list):
    """Print the column→device mapping for *house_number* and highlight
    which column each target appliance resolves to."""
    if house_number not in HOUSE_APPLIANCES:
        return
    house_map = {
        f"Appliance{i+1}": name
        for i, name in enumerate(HOUSE_APPLIANCES[house_number])
    }
    print(f"\nHouse {house_number} appliance map:")
    for col, name in house_map.items():
        print(f"  {col}: {name}")

    print(f"\nTarget appliance → column mapping for House {house_number}:")
    for appliance in target_appliances:
        col = get_appliance_column(house_number, appliance)
        if col is not None:
            device_label = house_map.get(col, "?")
            print(f"  {appliance:20s} → {col} ({device_label})")
        else:
            print(f"  {appliance:20s} → [NOT FOUND in House {house_number}]")


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
    # Backward-compatible single-house argument
    parser.add_argument(
        "--house",
        help=(
            "Path to a REFIT house CSV used for both training and testing. "
            "Deprecated in favour of --train-house / --test-house; "
            "kept for backward compatibility."
        ),
    )
    # Cross-house arguments
    parser.add_argument(
        "--train-house",
        metavar="CSV",
        help=(
            "Path to the REFIT house CSV used for training HMMs "
            "(e.g. '../Processed_Data_CSV/House_9.csv'). "
            "If omitted, --house is used."
        ),
    )
    parser.add_argument(
        "--test-house",
        metavar="CSV",
        help=(
            "Path to the REFIT house CSV used for disaggregation/evaluation "
            "(e.g. '../Processed_Data_CSV/House_3.csv'). "
            "If omitted, --house (or --train-house) is used."
        ),
    )
    parser.add_argument(
        "--appliances", nargs="+", default=DEFAULT_APPLIANCES,
        metavar="APPLIANCE",
        help=(
            "Canonical appliance names to process. "
            f"Default: {DEFAULT_APPLIANCES}. "
            "Supported: kettle, microwave, fridge, tv, washing_machine, "
            "dishwasher, tumble_dryer, toaster, freezer, computer."
        ),
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
        help=(
            "Run in true NILM mode: estimate appliance states from the "
            "aggregate signal only (no sub-metering columns used)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N samples (useful for quick tests)."
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation."
    )
    parser.add_argument(
        "--n-states", type=int, default=None,
        help="Override number of HMM hidden states for all appliances."
    )
    parser.add_argument(
        "--fridge-states", type=int, default=None,
        help="Override number of HMM hidden states specifically for fridge."
    )
    parser.add_argument(
        "--sample-limit", type=int, default=50_000,
        help="Maximum training samples per appliance (default: 50000)."
    )
    parser.add_argument(
        "--plot-preprocessing", action="store_true",
        help="Generate raw vs preprocessed plots for train/test signals."
    )
    parser.add_argument(
        "--preprocessing-plot-limit", type=int, default=3000,
        help="Maximum samples to render in preprocessing plots."
    )
    parser.add_argument(
        "--detect-events", action="store_true",
        help="Detect state transitions and print multiple events per appliance."
    )
    parser.add_argument(
        "--events-per-appliance", type=int, default=5,
        help="Number of events to display and plot for each appliance."
    )
    parser.add_argument(
        "--event-window", type=int, default=120,
        help="Half-window size (in samples) around each detected event plot."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve train/test CSV paths
    train_csv = args.train_house or args.house
    test_csv = args.test_house or args.train_house or args.house

    if train_csv is None:
        parser.error(
            "Provide at least one of --house, --train-house, or --test-house."
        )

    for label, path in [("train", train_csv), ("test", test_csv)]:
        if not os.path.isfile(path):
            print(f"\n[ERROR] {label.capitalize()} file not found: {path!r}")
            print(
                "Please download the REFIT dataset and place the CSV files in "
                "Processed_Data_CSV/ — see Processed_Data_CSV/README.md for "
                "instructions."
            )
            sys.exit(1)

    train_house_number = parse_house_number(train_csv)
    test_house_number = parse_house_number(test_csv)
    appliances = args.appliances

    appliance_n_states = {}
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
        print("  Cross-house : YES — models trained on House "
              f"{train_house_number}, evaluated on House {test_house_number}")
    print("=" * 60)

    # Print appliance maps for both houses
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

        print("\n[DONE] Results available in the returned DataFrame.")
        print("       Plots saved to: plots/")
        print("       Models saved to: models/")
        return results


if __name__ == "__main__":
    main()
