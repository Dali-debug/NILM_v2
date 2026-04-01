"""
preprocessing.py
----------------
Preprocessing utilities for REFIT CSV data.

Includes:
  - load_refit_csv()   : load and parse a REFIT house CSV into a DataFrame
  - hampel_filter()    : outlier removal using the Hampel identifier
  - interpolate_missing() : fill NaN gaps with linear interpolation
  - preprocess_house() : full preprocessing pipeline for one house CSV
"""

import os
import re

import numpy as np
import pandas as pd


def _extract_house_number(filepath: str) -> str:
    """Extract house number from a filename like 'House_3.csv'."""
    name = os.path.basename(filepath)
    match = re.search(r"house[_-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"


def plot_preprocessing_signals(raw_df: pd.DataFrame,
                               processed_df: pd.DataFrame,
                               columns: list[str],
                               house_number: str,
                               plots_dir: str | None = None,
                               limit: int = 3000,
                               plot_tag: str = ""):
    """Save side-by-side raw vs preprocessed signal plots for selected columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plots_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_raw = raw_df.iloc[:limit]
    plot_processed = processed_df.iloc[:limit]
    x = range(len(plot_processed))

    tag = f"_{plot_tag}" if plot_tag else ""

    for col in columns:
        if col not in plot_raw.columns or col not in plot_processed.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, plot_raw[col].values, color="gray", linewidth=0.8,
                alpha=0.8, label="Raw")
        ax.plot(x, plot_processed[col].values, color="teal", linewidth=1.0,
                alpha=0.9, label="Preprocessed")
        ax.set_title(
            f"House {house_number} preprocessing{tag}: {col}"
        )
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Power (W)")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")

        safe_col = col.lower().replace(" ", "_")
        out_path = os.path.join(
            plots_dir,
            f"house{house_number}{tag}_{safe_col}_preprocessing.png",
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  Preprocessing plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_refit_csv(filepath: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load a REFIT house CSV file and return a clean DataFrame.

    The function:
    1. Reads the CSV (handles both `Time` and `Unix` timestamp columns).
    2. Parses the `Time` column as a DatetimeIndex.
    3. Sorts by time and drops exact duplicates.
    4. Sets the DatetimeIndex.

    Parameters
    ----------
    filepath : str
        Path to the REFIT CSV, e.g. ``Processed_Data_CSV/House_3.csv``.
    max_rows : int or None
        If provided, read at most this many raw CSV rows.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with columns:
        ``[Aggregate, Appliance1, ..., Appliance9]``.
    """
    if max_rows is not None:
        df = pd.read_csv(filepath, nrows=max_rows)
    else:
        df = pd.read_csv(filepath)

    # Identify the time column (REFIT uses "Time" or unnamed first column)
    time_col = None
    for candidate in ["Time", "time", "timestamp", "Timestamp"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        # Fallback: first column
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col).drop_duplicates(subset=[time_col])
    df = df.set_index(time_col)

    # Drop the Unix column if present (not needed after indexing)
    for col in ["Unix", "unix", "UNIX"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace negative values with NaN (sensor errors)
    df = df.clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Hampel filter (outlier removal)
# ---------------------------------------------------------------------------

def hampel_filter(series: pd.Series, window_size: int = 15, n_sigmas: float = 3.0):
    """Apply the Hampel identifier to remove outliers from a power series.

    For each point the local median and MAD are computed over a symmetric
    window of half-width *window_size*.  Points deviating more than
    ``n_sigmas × k × MAD`` from the median are replaced by the median.
    ``k = 1.4826`` is the consistency factor for a Gaussian distribution.

    Parameters
    ----------
    series : pd.Series
        1-D time series of power readings (Watts).
    window_size : int
        Half-width of the sliding window (total window = 2*window_size + 1).
    n_sigmas : float
        Threshold multiplier.

    Returns
    -------
    filtered : pd.Series
        Series with outliers replaced by local medians.
    outlier_mask : pd.Series of bool
        Boolean Series; True where an outlier was detected.
    """
    k = 1.4826  # consistency constant

    rolling = series.rolling(window=2 * window_size + 1, center=True, min_periods=1)
    median = rolling.median()
    mad = rolling.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

    threshold = n_sigmas * k * mad
    outlier_mask = np.abs(series - median) > threshold

    filtered = series.copy()
    filtered[outlier_mask] = median[outlier_mask]

    return filtered, outlier_mask


# ---------------------------------------------------------------------------
# Interpolation of missing values
# ---------------------------------------------------------------------------

def interpolate_missing(series: pd.Series, method: str = "linear",
                        max_gap: int = 5) -> pd.Series:
    """Fill NaN values in *series* using pandas interpolation.

    Parameters
    ----------
    series : pd.Series
    method : str
        Pandas interpolation method: ``"linear"``, ``"polynomial"``,
        ``"spline"``, etc.  Defaults to ``"linear"``.
    max_gap : int
        Maximum consecutive NaN run to fill.  Larger gaps are left as NaN.

    Returns
    -------
    pd.Series
        Interpolated series.
    """
    if method in ("polynomial", "spline"):
        return series.interpolate(method=method, order=3, limit=max_gap,
                                  limit_direction="both")
    return series.interpolate(method=method, limit=max_gap,
                              limit_direction="both")


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_house(filepath: str,
                     hampel_window: int = 15,
                     hampel_sigmas: float = 3.0,
                     interp_method: str = "linear",
                     max_gap: int = 10,
                     max_rows: int | None = None,
                     resample_rule: str = "8s",
                     plot_preprocessing: bool = False,
                     preprocessing_plot_columns: list[str] | None = None,
                     preprocessing_plots_dir: str | None = None,
                     preprocessing_plot_limit: int = 3000,
                     preprocessing_plot_tag: str = "") -> pd.DataFrame:
    """Load and fully preprocess a REFIT house CSV.

    Steps
    -----
    1. Load CSV and parse timestamps.
    2. Resample to a regular 8-second grid using mean aggregation; this
       converts any sub-second or irregular readings to a uniform grid
       (gaps produce NaN, which are filled in step 4).
    3. Apply Hampel filter to every column to remove outlier spikes.
    4. Interpolate remaining NaN values.

    Parameters
    ----------
    filepath : str
        Path to the REFIT CSV file.
    hampel_window : int
        Half-width for the Hampel filter sliding window.
    hampel_sigmas : float
        Outlier detection threshold (number of MAD-based sigma).
    interp_method : str
        Pandas interpolation method for NaN filling.
    max_gap : int
        Maximum consecutive NaN run to interpolate.
    max_rows : int or None
        If provided, read at most this many rows from the CSV before
        preprocessing. Useful for fast smoke tests.
    resample_rule : str
        Pandas offset alias for resampling (``"8s"`` for REFIT's 8-second
        sampling rate).  Set to ``None`` to skip resampling.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with DatetimeIndex.
    """
    print(f"[preprocess] Loading {filepath} …")
    df = load_refit_csv(filepath, max_rows=max_rows)
    print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Resample to a regular grid
    if resample_rule:
        print(f"  Resampling to {resample_rule} grid …")
        df = df.resample(resample_rule).mean()

    raw_resampled = df.copy()

    # Apply Hampel filter + interpolation column by column
    print("  Applying Hampel filter and interpolation …")
    for col in df.columns:
        df[col], _ = hampel_filter(df[col], window_size=hampel_window,
                                   n_sigmas=hampel_sigmas)
        df[col] = interpolate_missing(df[col], method=interp_method,
                                      max_gap=max_gap)

    # Final safety: clip to non-negative and fill any remaining NaN with 0
    df = df.clip(lower=0).fillna(0)

    if plot_preprocessing:
        columns = preprocessing_plot_columns or ["Aggregate"]
        house_number = _extract_house_number(filepath)
        plot_preprocessing_signals(
            raw_df=raw_resampled,
            processed_df=df,
            columns=columns,
            house_number=house_number,
            plots_dir=preprocessing_plots_dir,
            limit=preprocessing_plot_limit,
            plot_tag=preprocessing_plot_tag,
        )

    print(f"  Preprocessing complete — {len(df):,} samples, "
          f"{df.isna().sum().sum()} NaN remaining.")
    return df
