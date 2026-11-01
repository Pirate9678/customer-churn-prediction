#!/usr/bin/env python3
"""Generate a large-scale synthetic telecom churn dataset.

The generator is calibrated on the real IBM Telco Customer Churn data:

  1. Feature vectors (X) are bootstrapped from the real data — sampling
     whole rows with replacement — so the joint distribution of customer
     attributes is preserved exactly.
  2. The churn label (y) is redrawn from P(churn | Contract, tenure-bucket,
     InternetService) estimated on the real data with Laplace smoothing,
     so the synthetic labels reproduce the real churn drivers
     (month-to-month contracts, short tenure and fibre-optic service
     churn the most) instead of inventing new ones.

The result is a high-volume dataset with realistic structure, suitable
for benchmarking the training pipeline at scale. Generation is seeded
and fully reproducible.

The large CSV is a build artifact (git-ignored); regenerate it with:

    python3 data/generate_large_dataset.py --rows 1200000

Honesty note: rows beyond the 7,043 real ones are synthetic. The README
states this, and the script prints the calibration summary so the
provenance is auditable.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import ID_COLUMN, TARGET_COLUMN, load_data, prepare_frame

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_DATA = os.path.join(ROOT, "data", "telco_customer_churn.csv")

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]
CELL_KEYS = ["Contract", "tenure_bucket", "InternetService"]
SMOOTHING = 5.0  # Laplace smoothing strength (in pseudo-observations)


def churn_cell_rates(df):
    """Estimate P(churn | Contract, tenure-bucket, InternetService).

    Returns (dict mapping cell -> smoothed rate, global rate, frame used).
    """
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    )
    global_rate = float(df[TARGET_COLUMN].mean())
    grouped = (
        df.groupby(CELL_KEYS, observed=True)[TARGET_COLUMN]
        .agg(["sum", "count"])
        .reset_index()
    )
    grouped["rate"] = (grouped["sum"] + SMOOTHING * global_rate) / (
        grouped["count"] + SMOOTHING
    )
    rates = {
        (row["Contract"], str(row["tenure_bucket"]), row["InternetService"]): float(
            row["rate"]
        )
        for _, row in grouped.iterrows()
    }
    return rates, global_rate, grouped


def generate(n_rows, seed=42, chunk=200_000, output_path=None):
    """Generate the synthetic dataset in chunks; returns the output path."""
    real_raw = load_data(REAL_DATA)
    column_order = list(real_raw.columns)
    feature_cols = [c for c in column_order if c not in (ID_COLUMN, TARGET_COLUMN)]

    real = prepare_frame(real_raw)
    rates, global_rate, grouped = churn_cell_rates(real)
    real["tenure_bucket"] = pd.cut(
        real["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    )
    print(f"Real rows: {len(real)}, global churn rate: {global_rate:.4f}")
    print(f"Calibration cells: {len(rates)} "
          f"(Contract x tenure-bucket x InternetService)")
    top = grouped.sort_values("rate", ascending=False).head(3)
    for _, r in top.iterrows():
        print(f"  highest-churn cell: {r['Contract']} / {r['tenure_bucket']} / "
              f"{r['InternetService']} -> {r['rate']:.3f} (n={int(r['count'])})")

    rng = np.random.default_rng(seed)
    real_features = real[feature_cols].reset_index(drop=True)
    real_cells = real[CELL_KEYS].reset_index(drop=True)

    if output_path is None:
        output_path = os.path.join(ROOT, "data", "telco_churn_large.csv")
    first = True
    made = 0
    n_chunks = (n_rows + chunk - 1) // chunk
    for ci in range(n_chunks):
        size = min(chunk, n_rows - made)
        idx = rng.integers(0, len(real_features), size=size)
        synth = real_features.iloc[idx].copy()
        cells = real_cells.iloc[idx]

        # Vectorised cell-rate lookup.
        key = list(
            zip(cells["Contract"], cells["tenure_bucket"].astype(str),
                cells["InternetService"])
        )
        proba = np.array([rates.get(k, global_rate) for k in key])
        churn = rng.binomial(1, proba)

        out = pd.DataFrame(index=range(made, made + size))
        out[ID_COLUMN] = [f"SYN-{i:07d}" for i in range(made, made + size)]
        for col in feature_cols:
            out[col] = synth[col].to_numpy()
        out[TARGET_COLUMN] = np.where(churn == 1, "Yes", "No")
        out = out[column_order]

        out.to_csv(output_path, mode="w" if first else "a",
                   header=first, index=False)
        first = False
        made += size
        print(f"  chunk {ci + 1}/{n_chunks}: {made:,}/{n_rows:,} rows written")

    print(f"Done -> {output_path}")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a large synthetic telco churn dataset "
                    "calibrated on the real IBM Telco data.")
    parser.add_argument("--rows", type=int, default=1_200_000,
                        help="Number of synthetic rows (default: 1200000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: data/telco_churn_large.csv)")
    args = parser.parse_args(argv)
    generate(args.rows, seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
