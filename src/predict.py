"""Predict churn for a single customer record.

Loads the fitted preprocessor and trained Random Forest from models/,
applies the same preprocessing used in training, and prints a JSON prediction.

Model artifacts are stored gzip-compressed (*.pkl.gz) to stay under GitHub's
per-file upload limit; plain *.pkl files are used automatically when present
(e.g. after running src/train.py locally).

Usage:
    # JSON record passed directly:
    python src/predict.py '{"gender": "Female", "SeniorCitizen": 0, ...}'

    # JSON record read from a file:
    python src/predict.py --file sample_customer.json

    # Batch: score every row of a CSV (raw feature columns, same names as
    # the dataset) and write the predictions alongside the input columns:
    python src/predict.py --batch customers.csv [--output scored.csv]

    The single-record CLI pays ~3s of interpreter + sklearn import and
    ~0.5s of artifact loading per call; the batch mode pays that once and
    scores thousands of rows in one vectorised transform/predict pass
    (~0.5s per 10k rows after load). Use it for any multi-record scoring.

The record must contain the raw feature columns (same names as the dataset,
without customerID/Churn). Missing columns are imputed by the pipeline;
unknown categorical values are ignored.
"""

import gzip
import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from features import FEATURE_COLUMNS, load_pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSOR_PATH = os.path.join(ROOT, "models", "preprocessor.pkl")
MODEL_PATH = os.path.join(ROOT, "models", "random_forest_churn.pkl")


def load_artifact(path):
    """Load a pickle artifact, transparently supporting a .pkl.gz fallback."""
    if os.path.exists(path):
        return load_pickle(path)
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rb") as fh:
            return pickle.load(fh)
    raise FileNotFoundError(
        f"model artifact not found: looked for {path} and {gz_path}. "
        "Run src/train.py to generate them."
    )


def risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def predict_frame(frame: pd.DataFrame, preprocessor, model) -> pd.DataFrame:
    """Score a whole frame at once (vectorised); returns the frame with
    churn_prediction / churn_probability / risk columns appended."""
    frame = frame.copy()
    for c in FEATURE_COLUMNS:
        if c not in frame.columns:
            frame[c] = None
    Xt = preprocessor.transform(frame[FEATURE_COLUMNS])
    proba = model.predict_proba(Xt)[:, 1]
    frame["churn_probability"] = [round(float(p), 4) for p in proba]
    frame["churn_prediction"] = (proba >= 0.5).astype(int)
    frame["churn_label"] = ["Yes" if p else "No"
                            for p in frame["churn_prediction"]]
    frame["risk"] = [risk_label(float(p)) for p in proba]
    return frame


def predict_record(record: dict) -> dict:
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    model = load_artifact(MODEL_PATH)

    frame = predict_frame(
        pd.DataFrame([{c: record.get(c) for c in FEATURE_COLUMNS}]),
        preprocessor, model)
    return {
        "churn_prediction": int(frame["churn_prediction"].iloc[0]),
        "churn_label": frame["churn_label"].iloc[0],
        "churn_probability": float(frame["churn_probability"].iloc[0]),
        "risk": frame["risk"].iloc[0],
    }


def main(argv):
    if len(argv) >= 3 and argv[1] == "--batch":
        in_path = argv[2]
        out_path = None
        if len(argv) == 5 and argv[3] == "--output":
            out_path = argv[4]
        elif len(argv) != 3:
            print(__doc__)
            sys.exit(1)
        if out_path is None:
            base, ext = os.path.splitext(in_path)
            out_path = f"{base}_scored{ext or '.csv'}"
        import time
        t0 = time.perf_counter()
        preprocessor = load_artifact(PREPROCESSOR_PATH)
        model = load_artifact(MODEL_PATH)
        load_s = time.perf_counter() - t0
        frame = pd.read_csv(in_path)
        t1 = time.perf_counter()
        scored = predict_frame(frame, preprocessor, model)
        score_s = time.perf_counter() - t1
        scored.to_csv(out_path, index=False)
        print(f"scored {len(scored):,} rows -> {out_path} "
              f"(load {load_s:.1f}s, score {score_s:.2f}s)")
        return
    if len(argv) == 3 and argv[1] == "--file":
        with open(argv[2]) as f:
            record = json.load(f)
    elif len(argv) == 2:
        record = json.loads(argv[1])
    else:
        print(__doc__)
        sys.exit(1)
    print(json.dumps(predict_record(record), indent=2))


if __name__ == "__main__":
    main(sys.argv)
