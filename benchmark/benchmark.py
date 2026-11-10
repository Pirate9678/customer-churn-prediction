#!/usr/bin/env python3
"""
Benchmark the optimised churn pipeline (Nov 2026).

Measures, end to end on this host:
  * data/generate_large_dataset.py  (1.2M synthetic rows; output must be
    byte-identical to the committed large CSV, md5-checked)
  * src/train.py                    (7k baseline, full GridSearchCV run)
  * src/train_large.py              (1M-row run; wall time + fit_seconds
    from models/metrics_large.json)
  * src/predict.py                  (single-record cold start; output must
    still be the README demo result, 0.9087 / High)
  * src/predict.py --batch          (new: score 10k rows in one process)

"Before" numbers are the pre-optimisation scripts measured 10 Nov 2026 on
this 4-vCPU host (same machine, same data sizes); they are recorded here
because the old scripts are superseded in the repo. "After" numbers are
measured live by this script (median of the stated reps).

The config-sweep experiment that chose the large model's parameters
(50 trees + max_samples=0.5) is recorded in results.json under
"experiments" — it was run once, 10 Nov 2026, with the setup (load/split/
preprocess) shared across configs, so it is not re-run here.

  .venv/bin/python benchmark/benchmark.py          # ~6 min, writes results.json
  .venv/bin/python benchmark/benchmark.py --quick  # generator + predict only, ~1 min

1.2M-row intermediates live in benchmark/tmp/ and are deleted afterwards;
only benchmark/results.json is kept.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "benchmark" / "tmp"
RESULTS = ROOT / "benchmark" / "results.json"

# Pre-optimisation timings (seconds), measured 10 Nov 2026 on this 4-vCPU
# host with the old scripts at the same data sizes (single clean runs for
# the generator and the two training scripts; median of 3 for predict).
# train_large fit_seconds is from the baseline re-run's metrics_large.json
# (the original 1 Nov 2026 run measured 83.9s on the same host — machine
# variance is ~10% at this scale, so compare the fit columns, not the wall).
BASELINE = {
    "generate_1m2": 9.6,
    "train_7k_wall": 76.1,
    "train_large_wall": 117.8,
    "train_large_fit": 92.5,
    "predict_cold": 3.5,
}

# Config sweep (benchmark/tmp/experiment_rf.py, 10 Nov 2026): one shared
# 1M-train / 200k-test setup, then a fit per config. The 100-tree baseline
# is the pre-optimisation large model.
EXPERIMENTS = {
    "setup_note": "shared 1M/200k split + preprocessing, 15.8s; "
                  "fit times only below",
    "configs": {
        "baseline_100trees": {
            "fit_s": 111.9, "accuracy": 0.7527, "precision": 0.5277,
            "recall": 0.7486, "f1": 0.6190, "roc_auc": 0.8252},
        "trees50": {
            "fit_s": 49.5, "accuracy": 0.7530, "precision": 0.5281,
            "recall": 0.7478, "f1": 0.6190, "roc_auc": 0.8252},
        "maxsamp0.5": {
            "fit_s": 89.7, "accuracy": 0.7524, "precision": 0.5273,
            "recall": 0.7482, "f1": 0.6187, "roc_auc": 0.8251},
        "trees50+maxsamp0.5_ADOPTED": {
            "fit_s": 32.9, "accuracy": 0.7523, "precision": 0.5271,
            "recall": 0.7483, "f1": 0.6185, "roc_auc": 0.8252},
        "depth10": {
            "fit_s": 86.2, "accuracy": 0.7532, "precision": 0.5284,
            "recall": 0.7481, "f1": 0.6193, "roc_auc": 0.8253},
        "hgb_REJECTED": {
            "fit_s": 92.5, "accuracy": 0.7826, "precision": 0.6348,
            "recall": 0.4472, "f1": 0.5247, "roc_auc": 0.8258,
            "note": "no faster than RF, and recall collapses 0.75 -> 0.45 "
                    "at the 0.5 threshold — wrong trade-off for retention"},
    },
}

DEMO_RECORD = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0, "TotalCharges": 190.5,
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, reps=2, check_stdout=None):
    """Run cmd `reps` times; return (median seconds, peak RSS MB, stdout).

    Peak RSS comes from resource.getrusage(RUSAGE_CHILDREN) — the max
    resident set of any child reaped so far, which is what we want when
    taking the max over reps. check_stdout, if given, is a function applied
    to each run's stdout that must return True.
    """
    import resource
    times, peak_rss, last_out = [], 0.0, ""
    for _ in range(reps):
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        out, _ = proc.communicate()
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(map(str, cmd))} failed "
                               f"(status {proc.returncode})")
        if check_stdout and not check_stdout(out):
            raise RuntimeError(f"output check failed for "
                               f"{' '.join(map(str, cmd))}:\n{out}")
        times.append(dt)
        peak_rss = max(peak_rss,
                       resource.getrusage(resource.RUSAGE_CHILDREN)
                       .ru_maxrss / 1024)  # KB -> MB on Linux
        last_out = out
    return statistics.median(times), peak_rss, last_out


def main():
    ap = argparse.ArgumentParser(description="Benchmark the pipeline.")
    ap.add_argument("--quick", action="store_true",
                    help="generator + predict only")
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    results = {"baseline": BASELINE, "experiments": EXPERIMENTS,
               "measured": {}}
    m = results["measured"]
    try:
        # ---- generator (1.2M rows) ------------------------------------
        gen_csv = TMP / "gen_1m2.csv"
        committed_md5 = md5(ROOT / "data" / "telco_churn_large.csv")

        def gen_check(_out):
            return md5(gen_csv) == committed_md5

        t, rss, _ = run([sys.executable,
                         str(ROOT / "data" / "generate_large_dataset.py"),
                         "--rows", "1200000", "--output", str(gen_csv)],
                        reps=3, check_stdout=gen_check)
        m["generate_1m2"] = {"s": round(t, 2), "peak_rss_mb": round(rss),
                             "output_md5_matches_committed_csv": True}
        print(f"generate 1.2M: {t:.2f}s (was {BASELINE['generate_1m2']}s; "
              f"output byte-identical to committed CSV)")

        # ---- predict: single-record cold start ------------------------
        def demo_check(out):
            return json.loads(out)["churn_probability"] == 0.9087

        t, rss, _ = run([sys.executable, str(ROOT / "src" / "predict.py"),
                         json.dumps(DEMO_RECORD)],
                        reps=3, check_stdout=demo_check)
        m["predict_cold"] = {"s": round(t, 2), "peak_rss_mb": round(rss),
                             "demo_probability": 0.9087}
        print(f"predict cold:  {t:.2f}s (was {BASELINE['predict_cold']}s; "
              f"demo record still 0.9087 / High)")

        # ---- predict: batch 10k rows (new capability) -----------------
        batch_in = TMP / "batch_10k.csv"
        batch_out = TMP / "batch_10k_scored.csv"
        with open(ROOT / "data" / "telco_churn_large.csv") as f:
            header = f.readline()
            lines = [next(f) for _ in range(10_000)]
        batch_in.write_text(header + "".join(lines))

        def batch_check(out):
            return "scored 10,000 rows" in out

        t, rss, out = run([sys.executable,
                           str(ROOT / "src" / "predict.py"), "--batch",
                           str(batch_in), "--output", str(batch_out)],
                          reps=2, check_stdout=batch_check)
        load_s = float(re.search(r"load ([\d.]+)s", out).group(1))
        score_s = float(re.search(r"score ([\d.]+)s", out).group(1))
        m["predict_batch_10k"] = {
            "s": round(t, 2), "peak_rss_mb": round(rss),
            "load_s": load_s, "score_s": score_s,
            "note": "no baseline — new capability; one cold start "
                    "amortised over 10k rows"}
        print(f"predict batch 10k: {t:.2f}s wall "
              f"(load {load_s:.1f}s + score {score_s:.2f}s)")

        if not args.quick:
            # ---- train 7k (full grid search run) ----------------------
            def train7k_check(_out):
                met = json.loads((ROOT / "models" / "metrics.json")
                                 .read_text())
                return met["test_metrics"]["accuracy"] == 0.7622

            t, rss, _ = run([sys.executable,
                             str(ROOT / "src" / "train.py")],
                            reps=2, check_stdout=train7k_check)
            m["train_7k_wall"] = {"s": round(t, 2),
                                  "peak_rss_mb": round(rss),
                                  "metrics_unchanged": True}
            print(f"train 7k:      {t:.2f}s "
                  f"(was {BASELINE['train_7k_wall']}s; metrics unchanged)")

            # ---- train 1M ---------------------------------------------
            def train1m_check(_out):
                met = json.loads((ROOT / "models" / "metrics_large.json")
                                 .read_text())
                tm = met["test_metrics"]
                return (abs(tm["accuracy"] - 0.7523) < 0.001
                        and abs(tm["roc_auc"] - 0.8252) < 0.001)

            t, rss, _ = run([sys.executable,
                             str(ROOT / "src" / "train_large.py")],
                            reps=2, check_stdout=train1m_check)
            fit_s = json.loads((ROOT / "models" / "metrics_large.json")
                               .read_text())["fit_seconds"]
            m["train_large_wall"] = {"s": round(t, 2),
                                     "peak_rss_mb": round(rss)}
            m["train_large_fit"] = {"s": round(fit_s, 1)}
            print(f"train 1M wall: {t:.2f}s "
                  f"(was {BASELINE['train_large_wall']}s)")
            print(f"train 1M fit:  {fit_s:.1f}s "
                  f"(was {BASELINE['train_large_fit']}s)")

        RESULTS.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {RESULTS}")
        for key, base_key in (("generate_1m2", "generate_1m2"),
                              ("train_7k_wall", "train_7k_wall"),
                              ("train_large_wall", "train_large_wall"),
                              ("train_large_fit", "train_large_fit"),
                              ("predict_cold", "predict_cold")):
            if key in m:
                b, a = BASELINE[base_key], m[key]["s"]
                print(f"  {key}: {b}s -> {a}s ({b / a:.2f}x)")
    finally:
        for f in TMP.glob("*.csv"):
            f.unlink()
        print("Cleaned up benchmark/tmp/*.csv")


if __name__ == "__main__":
    main()
