"""
streaming_demo.py
-----------------
Simulates a real-time NILM stream using aggregate-only data from a REFIT house CSV.

Samples are fed one-by-one to WindowNILM, which accumulates them in a buffer
and triggers NILM inference whenever a complete window is ready.  No sub-metering
columns are read — this is true NILM.

Usage
-----
    python streaming_demo.py \\
        --house Processed_Data_CSV/House_3.csv \\
        --models-dir models/9 \\
        --appliances kettle microwave fridge tv \\
        --n-samples 500 \\
        --window-seconds 60 \\
        --window-mode tumbling \\
        --delay-ms 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, deque
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.train_hmm import load_models
from pipeline.disaggregate import disaggregate_nilm
from utils import get_logger

# Silence INFO noise emitted by pipeline modules during inference
for _name in ("pipeline.disaggregate", "pipeline.train_hmm",
               "pipeline.preprocessing", "preprocessing"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = get_logger(__name__)

# REFIT native sampling cadence
REFIT_SAMPLE_SECONDS: int = 8


# ---------------------------------------------------------------------------
# WindowNILM
# ---------------------------------------------------------------------------

class WindowNILM:
    """Accumulates aggregate power samples and runs NILM inference on windows.

    Parameters
    ----------
    house_number : int
        House number associated with the models (used only as a label).
    appliances : list of str
        Canonical appliance names (e.g. ``["kettle", "fridge"]``).
    models_dir : str
        Directory that contains the trained HMM JSON files.
    window_size : int
        Number of samples that form one inference window.
    window_mode : {"tumbling", "sliding"}
        ``"tumbling"`` — non-overlapping; buffer is cleared after each inference.
        ``"sliding"``  — inference fires at every new sample once the buffer
        holds at least *window_size* entries.
    """

    def __init__(
        self,
        house_number: int,
        appliances: list[str],
        models_dir: str,
        window_size: int,
        window_mode: str = "tumbling",
    ) -> None:
        self.house_number = house_number
        self.appliances = list(appliances)
        self.window_size = window_size
        self.window_mode = window_mode
        self._window_count = 0

        # Load models — warn and skip any appliance whose file is missing
        self.models = load_models(house_number, appliances, models_dir=models_dir)
        missing = [a for a in appliances if a not in self.models]
        if missing:
            logger.warning(
                "No trained model found for: %s — those appliances will be skipped.",
                missing,
            )
        self._active_appliances: list[str] = [a for a in appliances if a in self.models]

        if not self._active_appliances:
            logger.warning("No models loaded — inference will produce empty results.")

        # For sliding mode the deque auto-discards oldest samples
        _maxlen = window_size if window_mode == "sliding" else None
        self._values: deque[float] = deque(maxlen=_maxlen)
        self._timestamps: deque = deque(maxlen=_maxlen)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def samples_buffered(self) -> int:
        """Number of samples currently in the window buffer."""
        return len(self._values)

    @property
    def windows_completed(self) -> int:
        """Total number of windows for which inference has been run."""
        return self._window_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, value: float, timestamp=None) -> Optional[dict]:
        """Feed one aggregate power sample into the window.

        Parameters
        ----------
        value : float
            Observed aggregate power in Watts.
        timestamp : any
            Datetime or integer sample index associated with this reading.

        Returns
        -------
        dict or None
            Inference result dict when a complete window is ready;
            ``None`` while the buffer is still filling.
        """
        self._values.append(float(value) if not np.isnan(value) else 0.0)
        self._timestamps.append(timestamp)

        if len(self._values) < self.window_size:
            return None

        result = self._infer()

        if self.window_mode == "tumbling":
            self._values.clear()
            self._timestamps.clear()

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self) -> dict:
        """Run combinatorial NILM inference on the current window contents.

        Reuses ``disaggregate_nilm`` from ``pipeline.disaggregate`` for the
        per-sample combinatorial search (aggregate ≈ Σ state means).
        A majority vote over the window then yields one state per appliance.
        """
        self._window_count += 1
        values = list(self._values)
        timestamps = list(self._timestamps)

        # Build a minimal DataFrame with only the Aggregate column —
        # disaggregate_nilm accesses df["Aggregate"].values exclusively.
        window_df = pd.DataFrame({"Aggregate": values})

        disagg = disaggregate_nilm(
            window_df,
            self.models,
            self.house_number,
            self._active_appliances,
        )

        appliances_result: dict[str, dict] = {}
        for appliance in self._active_appliances:
            label_col = f"{appliance}_state_label"
            power_col = f"{appliance}_power_est"

            if label_col not in disagg.columns:
                continue

            state_seq = disagg[label_col].tolist()
            counter = Counter(state_seq)
            winning_state, winning_count = counter.most_common(1)[0]
            confidence = winning_count / len(state_seq)

            # Power estimate: mean of the HMM-mean power for the winning state
            if power_col in disagg.columns:
                winning_powers = [
                    disagg[power_col].iloc[i]
                    for i, s in enumerate(state_seq)
                    if s == winning_state
                ]
                power_est = float(np.mean(winning_powers)) if winning_powers else 0.0
            else:
                power_est = 0.0

            appliances_result[appliance] = {
                "state": winning_state,
                "power_w": power_est,
                "confidence": confidence,
            }

        return {
            "window_index": self._window_count,
            "t_start": timestamps[0],
            "t_end": timestamps[-1],
            "mean_aggregate": float(np.mean(values)),
            "appliances": appliances_result,
        }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# Fixed column inner widths (number of chars between │ delimiters)
_CW = (14, 10, 10, 16)  # Appareil │ État │ Puissance │ Confiance


def _fmt_ts(ts) -> str:
    """Compact timestamp string for display."""
    if ts is None:
        return "?"
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def _hline(left: str, mid: str, right: str, sep: str) -> str:
    return left + sep.join("─" * w for w in _CW) + right


def _row(*cells: str) -> str:
    return "│" + "│".join(
        f"{c:<{w}}" for c, w in zip(cells, _CW)
    ) + "│"


def print_window_result(result: dict, appliances: list[str]) -> None:
    """Print a window inference result as a box-drawing table to stdout."""
    idx = result["window_index"]
    t0 = _fmt_ts(result["t_start"])
    t1 = _fmt_ts(result["t_end"])
    mean_agg = result["mean_aggregate"]

    inner_width = sum(_CW) + len(_CW) - 1  # column chars + inner separators
    header_line = f" Fenêtre #{idx} | {t0} → {t1} "
    power_line  = f" Puissance agrégée moyenne : {mean_agg:.1f} W "

    # Expand inner_width if header/power lines are longer
    inner_width = max(inner_width, len(header_line), len(power_line))

    top    = "┌" + "─" * inner_width + "┐"
    bottom = "└" + _hline("", "┴", "", "┴")[1:-0] + "┘"
    # Rebuild separators with exact column widths
    sep1   = _hline("├", "┬", "┤", "┬")
    sep2   = _hline("├", "┼", "┤", "┼")
    bot    = _hline("└", "┴", "┘", "┴")

    print(top)
    print("│" + header_line.ljust(inner_width) + "│")
    print("│" + power_line.ljust(inner_width) + "│")
    print(sep1)
    print(_row(" Appareil", " État", " Puissance", " Confiance"))
    print(sep2)

    for appliance in appliances:
        if appliance not in result["appliances"]:
            continue
        info = result["appliances"][appliance]
        state  = info["state"]
        power  = info["power_w"]
        conf   = info["confidence"]
        print(_row(
            f" {appliance.capitalize()}",
            f" {state}",
            f" {power:7.1f} W",
            f"  {conf:.0%}",
        ))

    print(bot)


# ---------------------------------------------------------------------------
# Shared CSV loader
# ---------------------------------------------------------------------------

def load_aggregate_series(csv_path: str, n_samples: int = 0) -> "pd.Series":
    """Load and return the Aggregate power series from a REFIT CSV.

    Handles time-column detection, index setting, numeric coercion, and
    NaN-to-zero replacement.  Raises ``ValueError`` if the Aggregate column
    is absent.

    Parameters
    ----------
    csv_path : str
        Path to a REFIT house CSV file.
    n_samples : int
        Maximum number of rows to read (0 = all).
    """
    df = pd.read_csv(csv_path, nrows=n_samples if n_samples > 0 else None)

    time_col: Optional[str] = next(
        (c for c in ("Time", "time", "timestamp", "Timestamp") if c in df.columns),
        None,
    )
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)

    if "Aggregate" not in df.columns:
        raise ValueError(f"'Aggregate' column not found in {csv_path!r}")

    series = pd.to_numeric(df["Aggregate"], errors="coerce").fillna(0.0)
    return series.iloc[:n_samples] if n_samples > 0 else series


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream(
    house_csv: str,
    models_dir: str,
    appliances: list[str],
    n_samples: int,
    window_seconds: int,
    window_mode: str,
    delay_ms: float,
    house_number: int,
    on_result=None,
    stop_event=None,
    nilm=None,
    on_sample=None,
) -> None:
    """Load aggregate data from *house_csv* and feed it through WindowNILM.

    Parameters
    ----------
    on_result : callable, optional
        Called with the result dict each time a complete window is inferred.
        Signature: ``on_result(result: dict) -> None``.
    stop_event : threading.Event, optional
        When set, the streaming loop exits at the next sample.
    nilm : WindowNILM, optional
        Existing engine to reuse.  When *None* a new one is created from
        *models_dir* / *appliances* / *window_seconds*.
    """
    window_size = max(1, window_seconds // REFIT_SAMPLE_SECONDS)

    print(f"CSV      : {house_csv}")
    print(f"Models   : {models_dir}")
    print(f"Samples  : {n_samples}  |  Window : {window_seconds}s = {window_size} samples  |  Mode : {window_mode}")
    print()

    # ── Load ONLY Aggregate + timestamp ──────────────────────────────────
    if not __import__("os").path.isfile(house_csv):
        sys.exit(f"Error: CSV file not found: {house_csv!r}")

    try:
        aggregate_series = load_aggregate_series(house_csv, n_samples)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    print(f"Loaded {len(aggregate_series)} samples.  Firing {window_mode} windows every {window_size} samples.\n")

    # ── Initialise WindowNILM ─────────────────────────────────────────────
    if nilm is None:
        nilm = WindowNILM(
            house_number=house_number,
            appliances=appliances,
            models_dir=models_dir,
            window_size=window_size,
            window_mode=window_mode,
        )

    delay_s    = delay_ms / 1000.0
    window_eta = window_size * delay_ms / 1000.0

    if delay_s > 0:
        print(f"Delai    : {delay_ms:.0f} ms/sample  →  ~{window_eta:.0f} s par fenetre\n")
    else:
        print("Delai    : 0 ms (mode rapide)\n")

    # ── Per-sample streaming loop ─────────────────────────────────────────
    for sample_idx, (ts, value) in enumerate(aggregate_series.items()):
        if stop_event is not None and stop_event.is_set():
            break

        clean_val = float(value) if not pd.isna(value) else 0.0

        # Position of this sample inside the current partial window (1-indexed)
        if window_mode == "tumbling":
            pos_in_win = len(nilm._values) + 1          # buffer before push
        else:  # sliding — deque is maxlen=window_size, so after fill always full
            pos_in_win = min(len(nilm._values) + 1, window_size)

        # Print window header at the start of each new tumbling window
        if window_mode == "tumbling" and pos_in_win == 1:
            next_win = nilm._window_count + 1
            eta_str  = f" (~{window_eta:.0f} s)" if delay_s > 0 else ""
            print(f"Fenetre #{next_win} | Collecte en cours{eta_str} ...")

        ts_str = _fmt_ts(ts) if ts is not None else f"sample {sample_idx}"
        print(f"  {ts_str}  {clean_val:8.1f} W  ({pos_in_win}/{window_size})", flush=True)

        # Simulate real-time arrival: wait before pushing the sample
        if delay_s > 0:
            time.sleep(delay_s)

        result = nilm.push(clean_val, timestamp=ts)

        if on_sample is not None:
            on_sample(nilm.samples_buffered, nilm.windows_completed)

        if result is not None:
            print()
            print_window_result(result, appliances)
            print()
            if on_result is not None:
                on_result(result)

    total = nilm._window_count
    print(f"Streaming termine — {total} fenetre{'s' if total != 1 else ''} inferee{'s' if total != 1 else ''}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Simulate real-time NILM on a REFIT house CSV.\n"
            "Only the Aggregate column is consumed — no sub-metering used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--house", default="Processed_Data_CSV/House_3.csv",
        metavar="CSV",
        help="Path to the REFIT house CSV (Aggregate column only).",
    )
    p.add_argument(
        "--models-dir", default="models/9",
        metavar="DIR",
        help="Directory containing trained HMM JSON files.",
    )
    p.add_argument(
        "--appliances", nargs="+",
        default=["kettle", "microwave", "fridge", "tv"],
        metavar="NAME",
        help="Canonical appliance names to disaggregate.",
    )
    p.add_argument(
        "--n-samples", type=int, default=10_000,
        metavar="N",
        help="Max aggregate samples to stream (default: 10000 reads entire demo CSV).",
    )
    p.add_argument(
        "--window-seconds", type=int, default=60,
        metavar="S",
        help="Window duration in seconds (7 samples at 8 s/sample → 56 s ≈ 1 min).",
    )
    p.add_argument(
        "--window-mode", choices=["tumbling", "sliding"], default="tumbling",
        help="'tumbling': non-overlapping windows.  'sliding': one inference per sample.",
    )
    p.add_argument(
        "--delay-ms", type=float, default=8000.0,
        metavar="MS",
        help="Inter-sample delay in ms (default: 8000 = real-time REFIT cadence; 0 = instant).",
    )
    p.add_argument(
        "--house-number", type=int, default=3,
        metavar="N",
        help="House number of the input CSV (used for contextual logging).",
    )
    return p


if __name__ == "__main__":
    # UTF-8 box-drawing characters on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    args = _build_parser().parse_args()
    stream(
        house_csv=args.house,
        models_dir=args.models_dir,
        appliances=args.appliances,
        n_samples=args.n_samples,
        window_seconds=args.window_seconds,
        window_mode=args.window_mode,
        delay_ms=args.delay_ms,
        house_number=args.house_number,
    )
