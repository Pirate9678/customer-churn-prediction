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


def predict_record(record: dict) -> dict:
    preprocessor = load_artifact(PREPROCESSOR_PATH)
    model = load_artifact(MODEL_PATH)

    frame = pd.DataFrame([{c: record.get(c) for c in FEATURE_COLUMNS}])
    Xt = preprocessor.transform(frame)
    proba = float(model.predict_proba(Xt)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "churn_prediction": pred,
        "churn_label": "Yes" if pred else "No",
        "churn_probability": round(proba, 4),
        "risk": risk_label(proba),
    }


def main(argv):
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
