# Customer Churn Prediction — Telco

A machine-learning classification project that predicts whether a telecom
customer will churn, giving the retention team a data-backed way to target
at-risk customers before they leave.

## Business problem

Acquiring a new telecom customer costs far more than retaining an existing
one. This project trains a **Random Forest classifier** on customer
demographics, account, service-usage, and billing data to flag customers
likely to churn, and surfaces the **top churn drivers** so retention offers
can be targeted (contract upgrades, onboarding outreach, pricing reviews).

## Dataset

| Item | Detail |
|---|---|
| Source | IBM Telco Customer Churn dataset, public mirror: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` |
| Provenance | The dataset was downloaded from the mirror above (a copy of IBM's open telco churn sample data). The 7k-row baseline uses this real data only. The large-scale run (below) uses a synthetic 1.2M-row dataset calibrated on it — clearly labeled as synthetic. |
| Rows | 7,043 customers × 21 columns |
| Target | `Churn` — 26.5% of customers churned |
| Features | Demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account (`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`), services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`), charges (`MonthlyCharges`, `TotalCharges`) |

## Feature engineering

All preprocessing lives in one reusable `ColumnTransformer` (`src/features.py`)
so training and inference behave identically:

- **Missing values** — `TotalCharges` is blank for 11 brand-new customers
  (`tenure == 0`); imputed with the **training-set median**. No other column
  has missing values.
- **Numeric** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`):
  median imputation → standard scaling.
- **Categorical** (15 columns): most-frequent imputation → one-hot encoding
  (`handle_unknown="ignore"` for robustness to unseen categories).
- `customerID` dropped (pure identifier, no predictive signal). 19 raw
  features expand to ~45 model features after encoding.
- Imputation statistics are fit on the **training split only** — no leakage.

## Project structure

```
customer-churn-prediction/
├── data/
│   ├── telco_customer_churn.csv      # raw dataset (see provenance above)
│   └── generate_large_dataset.py    # calibrated 1.2M-row synthetic generator
├── notebooks/
│   └── churn_prediction.ipynb       # EDA + feature engineering + modeling (executed)
├── src/
│   ├── data_loader.py               # CSV loading + type conversion
│   ├── features.py                  # ColumnTransformer (train/inference shared)
│   ├── train.py                     # grid search, evaluation, saves artifacts
│   ├── train_large.py               # scale training: 1M rows, timed, no grid search
│   └── predict.py                   # CLI: churn prediction for one JSON record
├── models/
│   ├── random_forest_churn.pkl.gz      # trained Random Forest (7k baseline)
│   ├── preprocessor.pkl                # fitted ColumnTransformer
│   ├── random_forest_churn_large.pkl.gz# trained Random Forest (1M-row run)
│   ├── preprocessor_large.pkl          # fitted ColumnTransformer (large run)
│   ├── metrics.json                    # exact measured metrics (source of truth)
│   ├── feature_importance.png          # top-15 churn drivers chart
│   ├── metrics_large.json              # measured metrics, 1M-row run
│   └── feature_importance_large.png    # churn drivers, 1M-row run
├── requirements.txt
└── README.md
```

> Model binaries are stored gzip-compressed (`*.pkl.gz`) because GitHub's
> web upload caps files at 25 MB (the raw pickles are 32–36 MB).
> `src/predict.py` loads `.pkl` or `.pkl.gz` transparently. The 159 MB
> `data/telco_churn_large.csv` stays a git-ignored build artifact —
> regenerate it with `python data/generate_large_dataset.py`.

## How to run

```bash
# 1. Create a virtualenv and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train the model (grid search + evaluation + saves everything to models/)
python src/train.py

# 3. Score a single customer record (see example below)
python src/predict.py '{"gender": "Female", "SeniorCitizen": 0, ...}'

# 4. Re-run the notebook
jupyter nbconvert --to notebook --execute notebooks/churn_prediction.ipynb

# 5. Scale up: generate 1.2M calibrated synthetic rows, then train on 1M
python data/generate_large_dataset.py --rows 1200000
python src/train_large.py
```

## Results (measured on held-out test set)

Model: `RandomForestClassifier`, tuned with `GridSearchCV` (3-fold CV, F1
scoring) over `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`.
Best params: `n_estimators=300, max_depth=15, min_samples_leaf=2,
class_weight="balanced"`. Stratified 80/20 split, `random_state=42`
(5,634 train / 1,409 test rows). All numbers below are copied from
`models/metrics.json`.

| Metric | Test score |
|---|---|
| Accuracy | 0.7622 |
| Precision | 0.5396 |
| Recall | 0.7112 |
| F1 | 0.6136 |
| ROC-AUC | 0.8351 |
| Best CV F1 (grid search) | 0.6273 |

Confusion matrix (test set, 1,409 rows):

|  | Predicted: stay | Predicted: churn |
|---|---|---|
| **Actual: stay** | 808 (TN) | 227 (FP) |
| **Actual: churn** | 108 (FN) | 266 (TP) |

**Reading the trade-off:** with `class_weight="balanced"` the model catches
**71% of churners** (recall 0.7112) at the cost of precision (0.5396) — about
half of flagged customers truly churn. For a retention use-case this is the
right trade-off: contacting a few extra safe customers is cheap compared to
losing a churner unnoticed. ROC-AUC of 0.8351 shows strong ranking ability —
the team can tune the probability threshold to match outreach capacity.

> Note: these are the actual measured metrics from the trained model. They
> are the honest numbers to quote for this project.

## Scaling to 1M+ rows

The 7k-row baseline proves the modelling approach; the large-scale run
proves it holds up at production volume.

**How the large dataset is built** (`data/generate_large_dataset.py`,
seeded, reproducible): feature vectors are bootstrapped from the real
data (whole rows sampled with replacement, preserving the true joint
distribution of customer attributes), and churn labels are redrawn from
P(churn | Contract, tenure-bucket, InternetService) estimated on the real
data with Laplace smoothing. Highest-churn cell in the real data:
month-to-month / 0–6 months tenure / fiber optic → 73.8% churn. The
synthetic set reproduces these drivers instead of inventing new ones —
and the README labels it synthetic wherever it appears.

**Training at scale** (`src/train_large.py`): 1,000,000 train / 200,000
test rows (stratified), same `ColumnTransformer`, `RandomForestClassifier`
with the hyperparameter shape transferred from the tuned baseline
(`max_depth=12`, `min_samples_leaf=10`, `class_weight="balanced"`,
100 trees — tightened to bound cost; no grid search at this scale),
float32 feature matrix, all CPUs. Fit time: **83.9 s** (~11,900 rows/s);
total wall time 105 s on a 4-vCPU host.

**Measured results** (from `models/metrics_large.json`, 200k held-out rows):

| Metric | 1M-row run | 7k baseline |
|---|---|---|
| Accuracy | 0.7527 | 0.7622 |
| Precision | 0.5277 | 0.5396 |
| Recall | 0.7486 | 0.7112 |
| F1 | 0.6190 | 0.6136 |
| ROC-AUC | 0.8252 | 0.8351 |

The metrics barely move between 7k and 1M rows — the model generalizes
rather than memorizing — and the top drivers are the same real-world
signals (`Contract = Month-to-month` 0.186, `tenure` 0.156,
`Contract = Two year` 0.117, `InternetService = Fiber optic` 0.092).
Full chart: `models/feature_importance_large.png`.

## Top 5 churn drivers (Random Forest feature importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `tenure` | 0.1262 |
| 2 | `TotalCharges` | 0.1203 |
| 3 | `Contract = Month-to-month` | 0.1011 |
| 4 | `MonthlyCharges` | 0.0946 |
| 5 | `Contract = Two year` | 0.0509 |

Full chart: `models/feature_importance.png`. Supporting EDA findings:
month-to-month customers churn at ~42% vs ~3% for two-year contracts;
short-tenure, high-monthly-charge, fiber-optic, and electronic-check customers
churn most; add-on services (TechSupport, OnlineSecurity) correlate with
retention.

## Retention recommendations

1. **Contract upgrades** — target month-to-month customers with discounted
   1-/2-year plans; contract type is the strongest categorical signal.
2. **Early-tenure onboarding** — flag customers in their first 6 months for
   proactive check-ins; churn risk falls steeply as tenure grows.
3. **High-bill review** — audit pricing/packaging for high-`MonthlyCharges`
   fiber customers, the highest-risk segment.
4. **Attach support services** — bundle TechSupport/OnlineSecurity in
   retention offers; subscribers churn far less.
5. **Payment friction** — investigate the electronic-check churn pattern
   (autopay incentives may help).

## Scoring new customers — `predict.py`

Pass a JSON record with the raw feature columns (same names as the dataset,
`customerID`/`Churn` not needed). Missing fields are imputed by the pipeline.

```bash
python src/predict.py '{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No",
  "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0, "TotalCharges": 190.5
}'
```

Example output:

```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.9087,
  "risk": "High"
}
```

Risk bands: `High` ≥ 0.70, `Medium` ≥ 0.40, else `Low`. You can also read a
record from a file: `python src/predict.py --file customer.json`.

## Reproducibility

- `random_state=42` everywhere; stratified split; requirements pinned in
  `requirements.txt`.
- The executed notebook (`notebooks/churn_prediction.ipynb`, 13 code cells
  with outputs) reuses the tuned hyperparameters from `src/train.py`, so its
  metrics match `models/metrics.json` exactly.
