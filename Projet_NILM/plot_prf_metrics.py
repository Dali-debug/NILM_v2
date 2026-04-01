"""
plot_prf_metrics.py
-------------------
Compute precision/recall/F1 per appliance and save a grouped bar plot.

Example:
    python plot_prf_metrics.py \
        --train-house ../Processed_Data_CSV/House_9.csv \
        --test-house ../Processed_Data_CSV/House_3.csv \
        --limit 200
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from disaggregate import run_disaggregation
from preprocessing import preprocess_house
from refit_metadata import get_appliance_column, parse_house_number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot precision, recall, and F1-score for NILM outputs."
    )
    parser.add_argument("--train-house", required=True,
                        help="Training house CSV path.")
    parser.add_argument("--test-house", required=True,
                        help="Test house CSV path.")
    parser.add_argument("--appliances", nargs="+",
                        default=["kettle", "microwave", "fridge", "tv"])
    parser.add_argument("--limit", type=int, default=200,
                        help="Max CSV rows to read for quick evaluation.")
    parser.add_argument("--out", default=None,
                        help="Optional output PNG path.")
    return parser


def main():
    args = build_parser().parse_args()

    train_house_number = parse_house_number(args.train_house)
    test_house_number = parse_house_number(args.test_house)

    results = run_disaggregation(
        house_csv=args.test_house,
        target_appliances=args.appliances,
        nilm_mode=False,
        limit=args.limit,
        plot=False,
        train_house_number=train_house_number,
        detect_events=False,
    )

    df = preprocess_house(args.test_house, max_rows=args.limit)

    appliances = []
    precision_vals = []
    recall_vals = []
    f1_vals = []

    for appliance in args.appliances:
        state_col = f"{appliance}_state_label"
        if state_col not in results.columns:
            continue

        col = get_appliance_column(test_house_number, appliance)
        if col is None or col not in df.columns:
            continue

        gt_on = (df[col] > 10).astype(int)
        pred_on = (results[state_col] != "OFF").astype(int)

        common_idx = results.index.intersection(df.index)
        y_true = gt_on.loc[common_idx].values
        y_pred = pred_on.loc[common_idx].values

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        appliances.append(appliance)
        precision_vals.append(float(precision))
        recall_vals.append(float(recall))
        f1_vals.append(float(f1))

    if not appliances:
        raise RuntimeError("No appliance metrics available to plot.")

    x = np.arange(len(appliances))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision_vals, width=width, label="Precision")
    ax.bar(x, recall_vals, width=width, label="Recall")
    ax.bar(x + width, f1_vals, width=width, label="F1-score")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in appliances])
    ax.set_ylabel("Score")
    ax.set_title(
        f"PR/F1 per appliance (train House {train_house_number} -> "
        f"test House {test_house_number}, limit={args.limit})"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    for i, (p, r, f1) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
        ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=8)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(
            script_dir,
            "plots",
            f"prf_house{train_house_number}_to_house{test_house_number}_limit{args.limit}.png",
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

    print("\nPRF metrics:")
    for appliance, p, r, f1 in zip(appliances, precision_vals, recall_vals, f1_vals):
        print(f"  {appliance:10s} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")

    print(f"\nPlot saved -> {out_path}")


if __name__ == "__main__":
    main()
