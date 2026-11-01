"""Feature engineering for the churn model.

Defines a single ColumnTransformer used everywhere (training notebook,
train.py, predict.py) so preprocessing is identical at train and inference
time. The fitted transformer is serialized to models/preprocessor.pkl.

Numeric: median imputation + standard scaling.
Categorical: most-frequent imputation + one-hot encoding (unknown categories
are ignored so the API is robust to unseen values).
"""

import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the churn feature set."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Column names of the transformed matrix, in order (fitted only)."""
    cat_names = preprocessor.named_transformers_["cat"].named_steps[
        "onehot"
    ].get_feature_names_out(CATEGORICAL_FEATURES)
    return list(NUMERIC_FEATURES) + list(cat_names)


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
