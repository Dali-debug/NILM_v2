#!/usr/bin/env python3
"""
realtime_demo.py
----------------
Command-line demonstration of the real-time NILM inference engine.

The script feeds aggregate power values one-at-a-time through
:class:`pipeline.realtime.RealTimeNILM` and prints a JSON-like record per
sample.  Power values can come from:

* **interactive prompt** (type one value at a time)
* **stdin** (one float per line, Ctrl-D / EOF to stop)
* **a built-in synthetic test sequence** (``--demo`` flag)
* **a CSV file** with a column of aggregate power values (``--csv``)

Usage examples::

    # Synthetic demo (no data file needed)
    python realtime_demo.py --house 3 --demo

    # Pipe values from another process
    echo -e "200\\n250\\n1800\\n1820\\n220" | python realtime_demo.py --house 3

    # Interactive mode (type values, then 'q' to quit)
    python realtime_demo.py --house 3 --compact

    # Read from a CSV
    python realtime_demo.py --house 3 --csv path/to/House_3.csv --limit 50

    # Quiet: only show per-appliance state, no full JSON
    python realtime_demo.py --house 3 --demo --compact
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the Projet_NILM package root is on the path when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline.realtime import RealTimeNILM


# ---------------------------------------------------------------------------
# Demo power sequence
# ---------------------------------------------------------------------------

def _demo_sequence() -> list[float]:
    """Return a synthetic sequence that exercises all four appliances."""
    base = 200.0   # background load (Watts)
    seq: list[float] = []

    # Phase 1: only fridge cycling (~238 W compressor kick)
    seq += [base] * 20
    seq += [base + 238] * 15
    seq += [base] * 10

    # Phase 2: kettle on (~1830 W spike)
    seq += [base + 1830] * 12
    seq += [base] * 20

    # Phase 3: microwave on (~1000 W)
    seq += [base + 1000] * 10
    seq += [base] * 15

    # Phase 4: TV on (~120 W) + fridge cycling
    seq += [base + 120] * 20
    seq += [base + 120 + 238] * 10
    seq += [base + 120] * 10
    seq += [base] * 10

    # Add small Gaussian noise
    import numpy as np
    rng = np.random.default_rng(42)
    seq_arr = np.array(seq, dtype=float) + rng.normal(0, 8, len(seq))
    return seq_arr.clip(0).tolist()


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_full(result: dict) -> str:
    """Return a pretty-printed JSON string for one inference result."""
    return json.dumps(result, indent=2)


def _format_compact(result: dict) -> str:
    """Return a single-line summary for one inference result."""
    parts = [
        f"t={result['timestamp'][:19]}",
        f"raw={result['raw_power']:.1f}W",
        f"smooth={result['smoothed_power']:.1f}W",
    ]
    for name, info in result["appliances"].items():
        parts.append(
            f"{name}={info['state_label']}({info['estimated_power']:.0f}W,"
            f"conf={info['confidence']:.2f})"
        )
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Power-source iterators
# ---------------------------------------------------------------------------

def _iter_stdin() -> list[float]:
    """Read one float per line from stdin until EOF."""
    values: list[float] = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values.append(float(line))
        except ValueError:
            sys.stderr.write(f"[warn] Skipping non-numeric line: {line!r}\n")
    return values


def _run_interactive(rt: "RealTimeNILM", formatter) -> int:
    """Interactive prompt: infer and print immediately after each value entered.

    Parameters
    ----------
    rt : RealTimeNILM
        Already-initialised inference engine.
    formatter : callable
        ``_format_full`` or ``_format_compact``.

    Returns
    -------
    int
        Number of samples processed.
    """
    print("# Interactive mode: type one power value per line.", file=sys.stderr)
    print("# Type 'q' (or Ctrl-D) to stop.", file=sys.stderr)

    count = 0
    while True:
        try:
            line = input("power> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            break

        if not line:
            continue

        if line.lower() in {"q", "quit", "exit"}:
            break

        try:
            power = float(line)
        except ValueError:
            sys.stderr.write(f"[warn] Not a number: {line!r}\n")
            continue

        result = rt.update(aggregate_power=power)
        print(formatter(result), flush=True)
        count += 1

    return count


def _iter_csv(path: str, column: str, limit: int | None) -> list[float]:
    """Read aggregate power from a REFIT-style CSV file."""
    import pandas as pd
    df = pd.read_csv(path, usecols=[column], nrows=limit)
    return df[column].clip(lower=0).fillna(0).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time NILM demo: feed aggregate power values and "
                    "see per-appliance state estimates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--house", type=int, default=3,
        help="House number whose trained models to load (default: 3).",
    )
    parser.add_argument(
        "--appliances", nargs="+",
        default=["kettle", "microwave", "fridge", "tv"],
        help="Appliances to include (default: kettle microwave fridge tv).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on the built-in synthetic power sequence instead of stdin.",
    )
    parser.add_argument(
        "--csv", metavar="PATH",
        help="Read aggregate power from this REFIT CSV file.",
    )
    parser.add_argument(
        "--csv-column", default="Aggregate",
        help="Column name in the CSV file (default: Aggregate).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of samples to process.",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Print one compact line per sample instead of full JSON.",
    )
    parser.add_argument(
        "--input-buffer", type=int, default=None,
        help="Override input smoothing buffer size.",
    )
    parser.add_argument(
        "--state-buffer", type=int, default=None,
        help="Override per-appliance state stabilisation buffer size.",
    )
    parser.add_argument(
        "--hysteresis", type=float, default=None,
        help="Override hysteresis margin (Watts).",
    )
    parser.add_argument(
        "--transition-penalty", type=float, default=None,
        help="Override transition penalty (Watts).",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Path to directory containing <appliance>_hmm.json files.",
    )
    args = parser.parse_args()

    # Build kwargs for RealTimeNILM (only pass overrides if supplied)
    kwargs: dict = {}
    if args.input_buffer is not None:
        kwargs["input_buffer_size"] = args.input_buffer
    if args.state_buffer is not None:
        kwargs["state_buffer_size"] = args.state_buffer
    if args.hysteresis is not None:
        kwargs["hysteresis_margin"] = args.hysteresis
    if args.transition_penalty is not None:
        kwargs["transition_penalty"] = args.transition_penalty
    if args.models_dir is not None:
        kwargs["models_dir"] = args.models_dir

    # Instantiate the engine
    try:
        rt = RealTimeNILM(
            house_number=args.house,
            appliances=args.appliances,
            **kwargs,
        )
    except RuntimeError as exc:
        sys.stderr.write(f"[error] {exc}\n")
        sys.exit(1)

    print(f"# {rt}", file=sys.stderr)

    formatter = _format_compact if args.compact else _format_full

    # Interactive mode: infer and print immediately per entry, no buffering.
    if not args.demo and not args.csv and sys.stdin.isatty():
        count = _run_interactive(rt, formatter)
        print(
            f"\n# Processed {count} samples.  Warmed up: {rt.is_warmed_up}",
            file=sys.stderr,
        )
        return

    # Batch modes: collect all values first, then stream through inference.
    if args.demo:
        power_values = _demo_sequence()
        print("# Running built-in synthetic demo sequence …", file=sys.stderr)
    elif args.csv:
        power_values = _iter_csv(args.csv, args.csv_column, args.limit)
        print(f"# Reading from CSV: {args.csv}", file=sys.stderr)
    else:
        print("# Reading from stdin (one float per line, Ctrl-D to stop) …",
              file=sys.stderr)
        power_values = _iter_stdin()

    if args.limit is not None:
        power_values = power_values[: args.limit]

    if not power_values:
        sys.stderr.write("[warn] No power values received.\n")
        sys.exit(0)

    for pw in power_values:
        result = rt.update(aggregate_power=pw)
        print(formatter(result), flush=True)

    print(
        f"\n# Processed {len(power_values)} samples.  "
        f"Warmed up: {rt.is_warmed_up}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
