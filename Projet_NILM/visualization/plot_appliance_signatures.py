"""
plot_appliance_signatures.py
----------------------------
Plot appliance power signatures from REFIT house CSV files.

For each target appliance, this script builds a signature from active power
samples (power >= threshold) across one or more houses and saves:
  1) Overlaid density histograms per house
  2) Per-house boxplots of active power
  3) A CSV summary table of descriptive statistics

Usage examples
--------------
    # Default: all houses in ../Processed_Data_CSV for 4 common appliances
    python plot_appliance_signatures.py

    # Select appliances and limit rows per house
    python plot_appliance_signatures.py \\
        --appliances kettle microwave fridge tv \\
        --limit 100000

    # Select specific houses
    python plot_appliance_signatures.py \\
        --houses ../Processed_Data_CSV/House_3.csv ../Processed_Data_CSV/House_9.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import POWER_ON_THRESHOLD
from .plot_utils import save_fig, style_axes
from pipeline.preprocessing import load_refit_csv, preprocess_house
from data.refit_metadata import get_appliance_column, parse_house_number
from utils import get_logger, state_labels

logger = get_logger(__name__)

DEFAULT_APPLIANCES = ["kettle", "microwave", "fridge", "tv"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot appliance signatures from REFIT house CSV files."
    )
    parser.add_argument(
        "--houses", nargs="+", default=None,
        help=(
            "List of REFIT house CSV paths. If omitted, all files matching "
            "../Processed_Data_CSV/House_*.csv are used."
        ),
    )
    parser.add_argument(
        "--appliances", nargs="+", default=DEFAULT_APPLIANCES,
        help=f"Canonical appliance names. Default: {DEFAULT_APPLIANCES}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Read at most N rows per house CSV for faster plotting.",
    )
    parser.add_argument(
        "--min-power", type=float, default=POWER_ON_THRESHOLD,
        help=f"Minimum power (W) to consider a sample as active (default: {POWER_ON_THRESHOLD}).",
    )
    parser.add_argument(
        "--sample-per-house", type=int, default=30000,
        help="Maximum active samples kept per (house, appliance).",
    )
    parser.add_argument("--bins", type=int, default=80,
                        help="Number of bins for histogram plots.")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for plots and CSV summary.")
    parser.add_argument(
        "--plot-clustered-state-signature", action="store_true",
        help="Run K-Means clustering and plot the state signature.",
    )
    parser.add_argument("--state-house", type=int, default=None)
    parser.add_argument("--state-appliance", default=None)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--state-plot-limit", type=int, default=3000)
    return parser


def resolve_house_files(houses: list[str] | None, script_dir: str) -> list[str]:
    if houses:
        files = sorted(houses)
    else:
        pattern = os.path.join(script_dir, "..", "Processed_Data_CSV", "House_*.csv")
        files = sorted(glob.glob(pattern))
    return [os.path.abspath(p) for p in files if os.path.isfile(p)]


def downsample(values: np.ndarray, max_points: int,
               rng: np.random.Generator) -> np.ndarray:
    if len(values) <= max_points:
        return values
    idx = rng.choice(len(values), size=max_points, replace=False)
    return values[idx]


def cluster_power_states(series: np.ndarray,
                         n_states: int) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0.0).values
    unique_count = int(pd.Series(values).nunique())
    k = max(1, min(int(n_states), unique_count))

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_raw = model.fit_predict(values.reshape(-1, 1))
    centers_raw = model.cluster_centers_.flatten()

    order = np.argsort(centers_raw)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(lbl)] for lbl in labels_raw], dtype=int)
    centers = centers_raw[order]
    return labels, centers


def plot_clustered_state_signature(
    *,
    appliance: str,
    house_n: int,
    series: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    outdir: str,
    state_plot_limit: int,
    bins: int,
) -> tuple[str, str]:
    n_states = len(centers)
    names = state_labels(n_states)  # from utils — no more local _state_names()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    ax_time, ax_hist = axes

    n_show = min(len(series), max(200, state_plot_limit))
    x = np.arange(n_show)
    y = series[:n_show]
    s = labels[:n_show]

    ax_time.plot(x, y, color="0.55", linewidth=1.0, alpha=0.8, label="Power")
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_states, 3)))
    for st in range(n_states):
        idx = np.where(s == st)[0]
        if len(idx) == 0:
            continue
        ax_time.scatter(idx, y[idx], s=8, color=colors[st], alpha=0.75,
                        label=f"{names[st]}")
        ax_time.axhline(float(centers[st]), color=colors[st], linestyle="--",
                        linewidth=1.0, alpha=0.6)

    style_axes(ax_time,
               title=f"House {house_n} - {appliance.capitalize()} clustered states "
                     f"(first {n_show} samples)",
               xlabel="Sample index",
               ylabel="Power (W)")
    ax_time.legend(loc="upper right", fontsize=8, ncol=2)

    all_vals = pd.to_numeric(pd.Series(series), errors="coerce").fillna(0.0).values
    x_low = float(np.percentile(all_vals, 1))
    x_high = float(np.percentile(all_vals, 99))
    if x_high <= x_low:
        x_low, x_high = float(np.min(all_vals)), float(np.max(all_vals))

    for st in range(n_states):
        vals = all_vals[labels == st]
        vals = vals[(vals >= x_low) & (vals <= x_high)]
        if len(vals) == 0:
            continue
        ax_hist.hist(vals, bins=bins, density=True, histtype="step",
                     linewidth=1.4, alpha=0.9, color=colors[st],
                     label=f"{names[st]} (center={centers[st]:.1f}W)")

    style_axes(ax_hist,
               title="State signature density (p1-p99)",
               xlabel="Power (W)",
               ylabel="Density")
    ax_hist.set_xlim(x_low, x_high)
    ax_hist.legend(fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(outdir, f"state_signature_house{house_n}_{appliance}.png")
    save_fig(fig, fig_path, dpi=130)

    state_rows = []
    total = len(labels)
    for st in range(n_states):
        cnt = int(np.sum(labels == st))
        state_rows.append({
            "house": house_n,
            "appliance": appliance,
            "state_id": st,
            "state_name": names[st] if st < len(names) else f"STATE_{st}",
            "center_w": float(centers[st]),
            "count": cnt,
            "share_pct": (100.0 * cnt / total) if total else 0.0,
        })

    csv_path = os.path.join(outdir, f"state_signature_house{house_n}_{appliance}.csv")
    pd.DataFrame(state_rows).to_csv(csv_path, index=False)

    return fig_path, csv_path


def plot_aggregate_raw_vs_preprocessed(
    *,
    house_n: int,
    raw_aggregate: pd.Series,
    processed_aggregate: pd.Series,
    outdir: str,
    limit: int,
) -> str:
    common_idx = processed_aggregate.index
    raw_on_grid = raw_aggregate.reindex(common_idx)

    n_show = min(len(common_idx), max(200, limit))
    x = np.arange(n_show)
    raw_vals = pd.to_numeric(raw_on_grid.iloc[:n_show], errors="coerce").values
    proc_vals = pd.to_numeric(processed_aggregate.iloc[:n_show], errors="coerce").values

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, raw_vals, color="red", linewidth=0.9, alpha=0.8,
            label="Raw (before preprocessing)")
    ax.plot(x, proc_vals, color="teal", linewidth=1.1, alpha=0.95,
            label="Preprocessed")
    ax.legend(loc="upper right")
    style_axes(ax,
               title=f"House {house_n} - Aggregate raw vs preprocessed (same time grid)",
               xlabel="Sample index",
               ylabel="Power (W)")
    fig.tight_layout()

    out_path = os.path.join(outdir, f"aggregate_raw_vs_preprocessed_house{house_n}.png")
    save_fig(fig, out_path, dpi=130)
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    outdir = args.outdir or os.path.join(script_dir, "plots", "signatures")
    os.makedirs(outdir, exist_ok=True)

    house_files = resolve_house_files(args.houses, script_dir)
    if not house_files:
        raise FileNotFoundError(
            "No house CSV files found. Provide --houses or place files in "
            "../Processed_Data_CSV/House_*.csv"
        )

    logger.info("Found %d house files.", len(house_files))
    logger.info("Target appliances: %s", args.appliances)
    logger.info("Output directory: %s", outdir)

    rng = np.random.default_rng(42)
    signatures: dict[str, list[tuple[int, np.ndarray]]] = {
        app: [] for app in args.appliances
    }
    full_series_map: dict[tuple[int, str], np.ndarray] = {}
    aggregate_raw_map: dict[int, pd.Series] = {}
    aggregate_processed_map: dict[int, pd.Series] = {}
    rows: list[dict] = []

    for house_csv in house_files:
        house_n = parse_house_number(house_csv)
        logger.info("[House %d] Loading %s", house_n, house_csv)
        df = preprocess_house(house_csv, max_rows=args.limit)
        raw_df = load_refit_csv(house_csv, max_rows=args.limit)

        if "Aggregate" in raw_df.columns and "Aggregate" in df.columns:
            raw_agg = pd.to_numeric(raw_df["Aggregate"], errors="coerce")
            raw_agg = raw_agg.resample("8s").mean()
            aggregate_raw_map[house_n] = raw_agg
            aggregate_processed_map[house_n] = pd.to_numeric(
                df["Aggregate"], errors="coerce"
            )

        for appliance in args.appliances:
            col = get_appliance_column(house_n, appliance)
            if col is None or col not in df.columns:
                logger.info("  - %s: not available in House %d", appliance, house_n)
                continue

            series = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
            full_series_map[(house_n, appliance)] = series
            active = series[series >= args.min_power]

            if len(active) == 0:
                logger.info("  - %s: no active samples >= %.1fW", appliance, args.min_power)
                continue

            active_ds = downsample(active, args.sample_per_house, rng)
            signatures[appliance].append((house_n, active_ds))

            rows.append({
                "house": house_n,
                "appliance": appliance,
                "column": col,
                "n_total": int(len(series)),
                "n_active": int(len(active)),
                "mean_w": float(np.mean(active)),
                "median_w": float(np.median(active)),
                "p90_w": float(np.percentile(active, 90)),
                "max_w": float(np.max(active)),
            })
            logger.info(
                "  - %s: %d active samples (col=%s)", appliance, len(active), col
            )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("No active appliance samples found to build signatures.")

    summary_path = os.path.join(outdir, "appliance_signature_summary.csv")
    summary_df.sort_values(["appliance", "house"]).to_csv(summary_path, index=False)
    logger.info("Saved summary CSV -> %s", summary_path)

    for appliance in args.appliances:
        by_house = sorted(signatures.get(appliance, []), key=lambda x: x[0])
        if not by_house:
            logger.info("[skip plot] No data for appliance '%s'", appliance)
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_hist, ax_box = axes

        combined = np.concatenate([vals for _, vals in by_house])
        x_low = float(np.percentile(combined, 1))
        x_high = float(np.percentile(combined, 99))
        if x_high <= x_low:
            x_low, x_high = float(np.min(combined)), float(np.max(combined))

        for house_n, vals in by_house:
            vals_hist = vals[(vals >= x_low) & (vals <= x_high)]
            if len(vals_hist) == 0:
                continue
            ax_hist.hist(vals_hist, bins=args.bins, density=True, alpha=0.8,
                         histtype="step", linewidth=1.2, label=f"House {house_n}")

        style_axes(ax_hist,
                   title=f"{appliance.capitalize()} signature density (p1-p99)",
                   xlabel="Active power (W)",
                   ylabel="Density")
        ax_hist.set_xlim(x_low, x_high)
        ax_hist.legend(fontsize=8)

        box_vals = [vals for _, vals in by_house]
        box_labels = [f"H{house_n}" for house_n, _ in by_house]
        ax_box.boxplot(box_vals, labels=box_labels, showfliers=False)
        style_axes(ax_box,
                   title=f"{appliance.capitalize()} per-house spread",
                   xlabel="House",
                   ylabel="Active power (W)")

        fig.suptitle(
            f"Appliance signature - {appliance} (active >= {args.min_power:.1f}W)",
            fontsize=12,
        )
        fig.tight_layout()
        out_path = os.path.join(outdir, f"signature_{appliance}.png")
        save_fig(fig, out_path, dpi=120)

    if args.plot_clustered_state_signature:
        target_house = args.state_house
        target_appliance = args.state_appliance or args.appliances[0]

        if target_house is None:
            candidate = [h for h, a in full_series_map.keys() if a == target_appliance]
            if not candidate:
                raise RuntimeError(
                    f"No series found for appliance '{target_appliance}'."
                )
            target_house = sorted(candidate)[0]

        key = (int(target_house), target_appliance)
        if key not in full_series_map:
            available = sorted([f"H{h}:{a}" for h, a in full_series_map.keys()])
            raise RuntimeError(
                f"No series found for House {target_house}, appliance "
                f"'{target_appliance}'. Available: {available}"
            )

        series = full_series_map[key]
        labels_arr, centers = cluster_power_states(series, args.n_states)
        fig_path, csv_path = plot_clustered_state_signature(
            appliance=target_appliance,
            house_n=int(target_house),
            series=series,
            labels=labels_arr,
            centers=centers,
            outdir=outdir,
            state_plot_limit=args.state_plot_limit,
            bins=args.bins,
        )
        logger.info("Saved clustered state signature -> %s", fig_path)
        logger.info("Saved clustered state summary -> %s", csv_path)

        h = int(target_house)
        if h in aggregate_raw_map and h in aggregate_processed_map:
            agg_path = plot_aggregate_raw_vs_preprocessed(
                house_n=h,
                raw_aggregate=aggregate_raw_map[h],
                processed_aggregate=aggregate_processed_map[h],
                outdir=outdir,
                limit=args.state_plot_limit,
            )
            logger.info("Saved aggregate raw vs preprocessed -> %s", agg_path)


if __name__ == "__main__":
    main()
