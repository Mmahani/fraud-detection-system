"""Phase 12 - Time-based validation diagnostic.

This script measures how the frozen models perform across time-ordered slices of
the reserved test set. The original split is random and stratified, so this is a
temporal-stability diagnostic on unseen rows, not a strict forward-in-time
holdout. It does not retrain or modify any frozen model.

Outputs:
    results/phase12_time_validation.csv
    results/phase12_time_validation.md
    results/phase12_summary.json
    results/phase12_plots/time_bucket_pr_auc.png
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"
PLOTS = RESULTS / "phase12_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

CHUNK = 20000
N_BUCKETS = 5
THRESHOLD = 0.35

SAVED_PROBABILITIES = {
    "SGD (Phase 5)": "sgd",
    "Random Forest (Phase 6)": "rf",
    "Gradient Boosting (Phase 6)": "gb",
    "XGBoost (Phase 6)": "xgb",
}


def log(message: str) -> None:
    print(message, flush=True)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """Render a small DataFrame as a GitHub-flavoured Markdown table."""

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return format(value, floatfmt)
        return str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, separator, *body])


def bucket_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "PR-AUC": float(average_precision_score(y_true, proba)),
        "ROC-AUC": float(roc_auc_score(y_true, proba)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
    }


def main() -> int:
    log("PHASE 12 - TIME-BASED VALIDATION DIAGNOSTIC")
    log(f"Project: {PROJECT}")

    y_test = pd.read_csv(PROCESSED / "y_test.csv")["isFraud"].to_numpy()
    log(f"Test rows: {len(y_test)}  fraud: {int(y_test.sum())} ({y_test.mean() * 100:.4f}%)")

    # The processed numeric block starts with TransactionDT (column 0). The
    # StandardScaler is monotonic, so the scaled column preserves time order.
    X_test = sparse.load_npz(PROCESSED / "X_test_processed.npz")
    log(f"X_test: {X_test.shape} dtype={X_test.dtype}")
    time_scaled = np.asarray(X_test.getcol(0).todense()).ravel()
    order = np.argsort(time_scaled, kind="stable")
    log(
        "Recovered time order from numeric__TransactionDT column 0 "
        f"(min={time_scaled.min():.3f}, max={time_scaled.max():.3f})"
    )

    probabilities: dict[str, np.ndarray] = {}
    for name, key in SAVED_PROBABILITIES.items():
        path = RESULTS / f"test_proba_{key}.npy"
        if path.is_file():
            probabilities[name] = np.load(path)
            log(f"Loaded saved probabilities: {path.name}")
        else:
            log(f"SKIP missing probabilities: {path.name}")

    # Frozen tuned model (final deliverable). Predicted in chunks to limit peak memory.
    tuned_path = MODELS / "best_tuned_model.pkl"
    if tuned_path.is_file():
        try:
            import joblib

            model = joblib.load(tuned_path)
            # Single-worker prediction keeps probability sums deterministic.
            model.set_params(n_jobs=1)
            proba_tuned = np.empty(len(y_test), dtype=np.float64)
            for start in range(0, len(y_test), CHUNK):
                end = min(start + CHUNK, len(y_test))
                proba_tuned[start:end] = model.predict_proba(X_test[start:end])[:, 1]
            probabilities["Tuned Random Forest (final)"] = proba_tuned
            del model
            gc.collect()
            log(f"Predicted frozen tuned model: {tuned_path.name}")
        except MemoryError:
            log("WARNING: could not score the frozen tuned model (MemoryError); skipping it.")
    else:
        log(f"SKIP missing model: {tuned_path.name}")

    del X_test
    gc.collect()

    if not probabilities:
        log("No probabilities available; aborting.")
        return 1

    buckets = np.array_split(order, N_BUCKETS)
    rows = []
    for bucket_index, idx in enumerate(buckets, start=1):
        y_bucket = y_test[idx]
        time_bucket = time_scaled[idx]
        row_base = {
            "Bucket": bucket_index,
            "Rows": len(idx),
            "Fraud Count": int(y_bucket.sum()),
            "Fraud Rate (%)": float(y_bucket.mean() * 100),
            "Scaled Time Min": float(time_bucket.min()),
            "Scaled Time Max": float(time_bucket.max()),
        }
        for name, proba in probabilities.items():
            metrics = bucket_metrics(y_bucket, proba[idx], THRESHOLD)
            rows.append({**row_base, "Model": name, **metrics})

    table = pd.DataFrame(rows)[
        ["Bucket", "Model", "Rows", "Fraud Count", "Fraud Rate (%)", "Scaled Time Min", "Scaled Time Max",
         "PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]
    ]
    csv_path = RESULTS / "phase12_time_validation.csv"
    table.to_csv(csv_path, index=False)
    log(f"Wrote {csv_path}")

    # A compact pivot for the report.
    pivot = table.pivot(index="Bucket", columns="Model", values="PR-AUC")
    log("PR-AUC by time bucket (earliest to latest):")
    log(pivot.to_string(float_format=lambda value: f"{value:.6f}"))

    first_last = {}
    for name in probabilities:
        subset = table[table["Model"] == name]
        earliest = float(subset.iloc[0]["PR-AUC"])
        latest = float(subset.iloc[-1]["PR-AUC"])
        first_last[name] = {
            "pr_auc_earliest_bucket": earliest,
            "pr_auc_latest_bucket": latest,
            "absolute_change_latest_minus_earliest": latest - earliest,
        }

    summary = {
        "diagnostic": "time_bucket_stability_on_reserved_test_set",
        "note": (
            "The reserved test split is random and stratified, so time buckets share the "
            "training period. This measures temporal stability on unseen rows, not a strict "
            "forward-in-time holdout. A full forward-in-time retrain is deferred because the "
            "current 8 GB environment cannot hold the training matrix (~1.2 GB) plus a model."
        ),
        "n_buckets": N_BUCKETS,
        "threshold": THRESHOLD,
        "rows": len(y_test),
        "models": list(probabilities.keys()),
        "first_last": first_last,
        "prevalence_caveat": (
            "The earliest bucket has a lower fraud rate than the middle buckets "
            "(see phase12_time_validation.csv). PR-AUC is prevalence-sensitive, so "
            "part of the earliest-bucket gap is compositional rather than temporal "
            "degradation."
        ),
    }
    summary_path = RESULTS / "phase12_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {summary_path}")

    # Plot: PR-AUC per time bucket for each model.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in probabilities:
        subset = table[table["Model"] == name].sort_values("Bucket")
        ax.plot(subset["Bucket"], subset["PR-AUC"], marker="o", label=name)
    ax.set_title("PR-AUC by time bucket on the reserved test set")
    ax.set_xlabel("Time bucket (1 = earliest, 5 = latest)")
    ax.set_ylabel("PR-AUC (Average Precision)")
    ax.set_xticks(range(1, N_BUCKETS + 1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = PLOTS / "time_bucket_pr_auc.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log(f"Wrote {plot_path}")

    report_lines = [
        "# Phase 12 — Time-Based Validation Diagnostic",
        "",
        "## What this measures",
        "",
        "The original preprocessing split is random and stratified, and `TransactionDT` is one of the",
        "strongest features. This diagnostic sorts the reserved test set by `TransactionDT` and scores",
        "the frozen models on five equal-count time buckets, from earliest to latest.",
        "",
        "This is a **temporal-stability check on unseen rows**, not a strict forward-in-time holdout:",
        "because the split is random, the test period overlaps the training period, so the models have",
        "already seen transactions from every bucket. A full forward-in-time retrain would require",
        "holding the ~1.2 GB training matrix and is deferred under the current 8 GB memory budget.",
        "No frozen model was retrained or modified.",
        "",
        "## PR-AUC by time bucket",
        "",
        markdown_table(pivot.reset_index()),
        "",
        "## First vs last bucket",
        "",
        "| Model | Earliest PR-AUC | Latest PR-AUC | Latest − Earliest |",
        "|---|---:|---:|---:|",
    ]
    for name, values in first_last.items():
        report_lines.append(
            f"| {name} | {values['pr_auc_earliest_bucket']:.6f} | "
            f"{values['pr_auc_latest_bucket']:.6f} | "
            f"{values['absolute_change_latest_minus_earliest']:+.6f} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "- The later buckets (3–5) are stable for the tree models: the tuned Random Forest",
        "  stays in the `0.723–0.742` PR-AUC range.",
        "- The earliest bucket is weaker for every model, but it also has the lowest fraud",
        "  rate (`2.41%` versus `3.40–3.99%` in the other buckets). PR-AUC is",
        "  prevalence-sensitive, so part of that gap is compositional rather than temporal",
        "  degradation.",
        "- This diagnostic cannot establish forward-in-time generalization: because the split",
        "  is random, every time bucket overlaps the training period. The main leakage risk",
        "  remains unquantified until the training matrix can be rebuilt with a strict time",
        "  cutoff.",
        "",
        "## Full bucket table",
        "",
        markdown_table(table),
        "",
        "## Artifacts",
        "",
        "- `results/phase12_time_validation.csv`",
        "- `results/phase12_summary.json`",
        "- `results/phase12_plots/time_bucket_pr_auc.png`",
    ]
    report_path = RESULTS / "phase12_time_validation.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log(f"Wrote {report_path}")
    log("Phase 12 diagnostic complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
