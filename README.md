# NILM — Non-Intrusive Load Monitoring

Energy disaggregation using Gaussian Hidden Markov Models on the [REFIT dataset](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned).

---

## Full Architecture

```
NILM_v2/
│
├── README.md                            ← This file
├── presentation.md                      ← Detailed project presentation (FR)
├── requirements.txt                     ← Runtime dependencies (pip install -r)
├── requirements-dev.txt                 ← Dev/test dependencies
├── .gitignore
│
├── Processed_Data_CSV/                  ← Input data
│   ├── House_3.csv
│   └── House_9.csv
│
└── Projet_NILM/                         ← Main Python package
    │
    ├── __init__.py                      ← Public API re-exports
    ├── run_nilm.py                      ← CLI entry point (run this)
    ├── config.py                        ← All tunable parameters
    ├── utils.py                         ← Shared utilities (logging, helpers)
    │
    ├── pipeline/                        ── Core pipeline steps ──────────────
    │   ├── __init__.py
    │   ├── preprocessing.py             ← Load, resample, Hampel filter, interpolate
    │   ├── train_hmm.py                 ← Fit GaussianHMM per appliance (Baum-Welch)
    │   └── disaggregate.py             ← Viterbi decoding + state labelling
    │
    ├── visualization/                   ── All plotting code ─────────────────
    │   ├── __init__.py
    │   ├── plot_utils.py                ← Shared figure helpers (save, style)
    │   ├── plot_prf_metrics.py          ← Precision / Recall / F1 bar charts
    │   └── plot_appliance_signatures.py ← Appliance power signature plots
    │
    ├── data/                            ── Metadata & mappings ───────────────
    │   ├── __init__.py
    │   ├── refit_metadata.py            ← Appliance ↔ CSV column mapping API
    │   └── refit_metadata.yaml          ← Mapping data (loaded by above)
    │
    ├── models/                          ── Saved HMM models (JSON) ──────────
    │   ├── 1/
    │   │   ├── fridge_hmm.json
    │   │   ├── kettle_hmm.json
    │   │   ├── microwave_hmm.json
    │   │   └── tv_hmm.json
    │   ├── 3/
    │   │   └── (same four files)
    │   └── 9/
    │       └── (same four files)
    │
    ├── plots/                           ── Generated output plots (PNG) ──────
    │   ├── house3_fridge_states.png
    │   ├── house3_kettle_states.png
    │   ├── house3_microwave_states.png
    │   ├── house3_tv_states.png
    │   ├── house9_*.png
    │   └── signatures/
    │       ├── signature_fridge.png
    │       ├── signature_kettle.png
    │       ├── signature_microwave.png
    │       ├── signature_tv.png
    │       ├── appliance_signature_summary.csv
    │       └── state_signature_house3_kettle.*
    │
    └── tests/                           ── Unit tests (pytest) ───────────────
        ├── __init__.py
        ├── test_preprocessing.py
        ├── test_hmm.py
        └── test_mapping.py
```

---

## Module Dependency Map

```
run_nilm.py
    ├── config.py
    ├── utils.py
    ├── data/refit_metadata.py
    ├── pipeline/train_hmm.py
    │       ├── config.py
    │       ├── utils.py
    │       ├── pipeline/preprocessing.py
    │       └── data/refit_metadata.py
    └── pipeline/disaggregate.py
            ├── config.py
            ├── utils.py
            ├── pipeline/preprocessing.py
            ├── pipeline/train_hmm.py
            ├── data/refit_metadata.py
            └── visualization/plot_utils.py
```

---

## Pipeline Overview

```
REFIT CSV (House_N.csv)
        │
        ▼
pipeline/preprocessing.py   — resample 8 s, Hampel filter, interpolate NaN
        │
        ▼
pipeline/train_hmm.py        — fit one GaussianHMM per appliance (Baum-Welch)
        │                      → saved to models/<house>/appliance_hmm.json
        ▼
pipeline/disaggregate.py     — Viterbi decoding, semantic state labelling
        │                      (sub-metering mode OR pure NILM mode)
        ▼
visualization/               — Precision / Recall / F1 plots, signature plots
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests
```

---

## Usage

Run all commands from `NILM_v2/Projet_NILM/`.

### Full pipeline — train on House 9, evaluate on House 3

```bash
python run_nilm.py \
    --train-house ../Processed_Data_CSV/House_9.csv \
    --test-house  ../Processed_Data_CSV/House_3.csv \
    --fridge-states 2 --plot-preprocessing --detect-events
```

### Train only

```bash
python run_nilm.py --train-house ../Processed_Data_CSV/House_9.csv --mode train
```

### Pure NILM mode (aggregate signal only)

```bash
python run_nilm.py \
    --train-house ../Processed_Data_CSV/House_9.csv \
    --test-house  ../Processed_Data_CSV/House_3.csv \
    --mode disaggregate --nilm
```

### PRF metrics plot

```bash
python visualization/plot_prf_metrics.py \
    --train-house ../Processed_Data_CSV/House_9.csv \
    --test-house  ../Processed_Data_CSV/House_3.csv
```

### Run tests

```bash
python -m pytest tests/
```

---

## Key CLI Arguments (`run_nilm.py`)

| Argument               | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `--train-house`        | Path to training house CSV                                   |
| `--test-house`         | Path to test house CSV                                       |
| `--appliances`         | Appliances to process (default: kettle microwave fridge tv)  |
| `--mode`               | `train`, `disaggregate`, or `all` (default: `all`)           |
| `--nilm`               | Use aggregate signal only (pure NILM mode)                   |
| `--fridge-states N`    | Number of HMM states for fridge: `2` or `3`                  |
| `--plot-preprocessing` | Save raw vs. preprocessed signal plots                       |
| `--detect-events`      | Detect and visualise state transitions                       |
| `--limit N`            | Cap data to N samples (quick testing)                        |

---

## Appliance-to-Column Mapping

| Appliance | House 9      | House 3      |
|-----------|--------------|--------------|
| Fridge    | Appliance1   | Appliance2   |
| Kettle    | Appliance7   | Appliance9   |
| Microwave | Appliance6   | Appliance8   |
| TV        | Appliance5   | Appliance7   |
