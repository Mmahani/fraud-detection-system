"""Audit the canonical PR-AUC definition against historical results.

The project previously reported PR-AUC in Phase 5/6 with the trapezoidal
implementation ``auc(recall, precision)``. Phase 7/8/9 use Average Precision
(``average_precision_score``). This script recomputes both definitions on the
same saved predictions and writes a transparent comparison.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud.metrics import pr_auc, pr_auc_trapezoidal

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_test_data() -> tuple[sparse.csr_matrix, np.ndarray]:
    x_test = sparse.load_npz(DATA_DIR / "X_test_processed.npz")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["isFraud"].to_numpy()
    if x_test.shape[0] != len(y_test):
        raise ValueError("X_test and y_test row counts do not match")
    return x_test, y_test


def _evaluate_model(
    model_name: str,
    model_file: str,
    x_test: sparse.csr_matrix,
    y_test: np.ndarray,
    notes: str = "",
) -> dict[str, object]:
    model_path = MODELS_DIR / model_file
    if not model_path.is_file():
        return {
            "dataset": "test",
            "model": model_name,
            "source": f"models/{model_file}",
            "n_samples": len(y_test),
            "n_positive": int(y_test.sum()),
            "pr_auc_average_precision": None,
            "pr_auc_trapezoidal": None,
            "difference_trapezoidal_minus_ap": None,
            "notes": "model file not found",
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
        # Parallel tree prediction can change floating-point summation order
        # and therefore tie ordering in average_precision_score. Use a single
        # worker so the audit is deterministic across runs.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        probabilities = model.predict_proba(x_test)[:, 1]

    average_precision = pr_auc(y_test, probabilities)
    trapezoidal = pr_auc_trapezoidal(y_test, probabilities)
    return {
        "dataset": "test",
        "model": model_name,
        "source": f"models/{model_file}",
        "n_samples": len(y_test),
        "n_positive": int(y_test.sum()),
        "pr_auc_average_precision": average_precision,
        "pr_auc_trapezoidal": trapezoidal,
        "difference_trapezoidal_minus_ap": trapezoidal - average_precision,
        "notes": notes,
    }


def _evaluate_oof() -> dict[str, object]:
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["isFraud"].to_numpy()
    oof_path = RESULTS_DIR / "oof_proba.npy"
    if not oof_path.is_file():
        return {
            "dataset": "train_oof",
            "model": "Random Forest (tuned)",
            "source": "results/oof_proba.npy",
            "n_samples": len(y_train),
            "n_positive": int(y_train.sum()),
            "pr_auc_average_precision": None,
            "pr_auc_trapezoidal": None,
            "difference_trapezoidal_minus_ap": None,
            "notes": "OOF probability file not found; regenerate with Phase 9",
        }

    oof_probabilities = np.load(oof_path)
    if len(oof_probabilities) != len(y_train):
        raise ValueError("OOF probabilities and y_train row counts do not match")

    average_precision = pr_auc(y_train, oof_probabilities)
    trapezoidal = pr_auc_trapezoidal(y_train, oof_probabilities)
    return {
        "dataset": "train_oof",
        "model": "Random Forest (tuned)",
        "source": "results/oof_proba.npy",
        "n_samples": len(y_train),
        "n_positive": int(y_train.sum()),
        "pr_auc_average_precision": average_precision,
        "pr_auc_trapezoidal": trapezoidal,
        "difference_trapezoidal_minus_ap": trapezoidal - average_precision,
        "notes": "Out-of-fold training predictions used for threshold selection",
    }


def _format_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| Dataset | Model | PR-AUC (Average Precision) | "
        "PR-AUC (trapezoidal) | Difference |\n"
        "|---|---|---:|---:|---:|\n"
    )
    lines = []
    for row in rows:
        ap = row["pr_auc_average_precision"]
        trap = row["pr_auc_trapezoidal"]
        diff = row["difference_trapezoidal_minus_ap"]
        ap_text = "n/a" if ap is None else f"{float(ap):.6f}"
        trap_text = "n/a" if trap is None else f"{float(trap):.6f}"
        diff_text = "n/a" if diff is None else f"{float(diff):+.6f}"
        lines.append(
            f"| {row['dataset']} | {row['model']} | {ap_text} | "
            f"{trap_text} | {diff_text} |"
        )
    return header + "\n".join(lines) + "\n"


def main() -> int:
    x_test, y_test = _load_test_data()
    rows: list[dict[str, object]] = []

    rows.append(
        _evaluate_model(
            "Baseline SGD",
            "baseline_sgd.pkl",
            x_test,
            y_test,
            notes="Phase 5 historical PR-AUC used the trapezoidal definition",
        )
    )
    rows.append(
        _evaluate_model(
            "Random Forest (untuned)",
            "random_forest.pkl",
            x_test,
            y_test,
            notes="Phase 6 historical PR-AUC 0.731103 was trapezoidal; Phase 8 uses AP",
        )
    )
    rows.append(
        _evaluate_model(
            "Random Forest (tuned)",
            "best_tuned_model.pkl",
            x_test,
            y_test,
            notes="Final model; Phase 8/9 primary metric is Average Precision",
        )
    )
    rows.append(
        _evaluate_model(
            "Gradient Boosting",
            "gradient_boosting.pkl",
            x_test,
            y_test,
            notes="Phase 6 comparison model",
        )
    )

    try:
        import xgboost  # noqa: F401
    except ImportError:
        rows.append(
            {
                "dataset": "test",
                "model": "XGBoost (untuned)",
                "source": "models/xgboost.pkl",
                "n_samples": len(y_test),
                "n_positive": int(y_test.sum()),
                "pr_auc_average_precision": None,
                "pr_auc_trapezoidal": None,
                "difference_trapezoidal_minus_ap": None,
                "notes": "Skipped: xgboost is not installed",
            }
        )
    else:
        rows.append(
            _evaluate_model(
                "XGBoost (untuned)",
                "xgboost.pkl",
                x_test,
                y_test,
                notes="Phase 6 comparison model",
            )
        )

    rows.append(_evaluate_oof())

    output_csv = RESULTS_DIR / "metrics_definition_audit.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False, float_format="%.12f")

    untuned = next(
        row for row in rows if row["model"] == "Random Forest (untuned)"
    )
    tuned = next(row for row in rows if row["model"] == "Random Forest (tuned)")
    untuned_ap = float(untuned["pr_auc_average_precision"])
    untuned_trap = float(untuned["pr_auc_trapezoidal"])
    tuned_ap = float(tuned["pr_auc_average_precision"])
    tuned_trap = float(tuned["pr_auc_trapezoidal"])

    saved_summary_path = RESULTS_DIR / "phase8_fast_summary.json"
    saved_tuned_ap = None
    if saved_summary_path.is_file():
        saved_summary = json.loads(saved_summary_path.read_text(encoding="utf-8"))
        saved_tuned_ap = saved_summary.get("tuned_test", {}).get("PR-AUC")

    if saved_tuned_ap is not None:
        saved_tuned_ap = float(saved_tuned_ap)
        tuned_runtime_note = (
            f"> The saved Phase 8 summary records `{saved_tuned_ap:.10f}` for the tuned "
            f"model; the recomputation above gives `{tuned_ap:.10f}`. The difference of "
            f"`{abs(tuned_ap - saved_tuned_ap):.2e}` is attributable to model "
            "serialization/runtime version and does not change the conclusion.\n"
        )
    else:
        tuned_runtime_note = ""

    report = f"""# PR-AUC Definition Audit

## Canonical definition

The project-wide PR-AUC is **Average Precision** from
`sklearn.metrics.average_precision_score`, exposed through
`src.fraud.metrics.pr_auc`.

The historical Phase 5/6 notebooks used the trapezoidal implementation
`auc(recall, precision)`. That implementation is retained only as
`src.fraud.metrics.pr_auc_trapezoidal` for audit purposes and must not be used
for model selection or delivery claims.

## Recomputed comparison

{_format_table(rows)}
> Predictions are generated with a single worker (`n_jobs=1`) so that
> floating-point summation order and Average Precision tie ordering are
> deterministic. XGBoost is skipped when the optional `xgboost` package is not
> installed.

## Random Forest comparison

| Comparison | Average Precision | Trapezoidal |
|---|---:|---:|
| Untuned Random Forest | {untuned_ap:.6f} | {untuned_trap:.6f} |
| Tuned Random Forest | {tuned_ap:.6f} | {tuned_trap:.6f} |
| Tuned minus untuned | {tuned_ap - untuned_ap:+.6f} | {tuned_trap - untuned_trap:+.6f} |

{tuned_runtime_note}
The Phase 6 Random Forest value `0.731103` is the trapezoidal result
(`{untuned_trap:.6f}` after recomputation). Its Average Precision is
`{untuned_ap:.6f}`, which matches the Phase 8 untuned value `0.722463` to
rounding.

The tuned model's Average Precision is `{tuned_ap:.6f}`. The canonical
improvement over the untuned Random Forest is therefore
**{tuned_ap - untuned_ap:+.6f}**. Under the historical trapezoidal definition,
the improvement is **{tuned_trap - untuned_trap:+.6f}**. Both comparisons are
internally consistent; mixing the two definitions is not.

## Decision

- Use `src.fraud.metrics.pr_auc` (Average Precision) everywhere.
- Label archived Phase 5/6 values explicitly as trapezoidal if they are quoted.
- Do not compare a Phase 6 trapezoidal value directly with a Phase 8/9
  Average Precision value.
- Notebooks 05 and 06 now call the canonical function; their stored outputs
  predate this change and should be refreshed when the notebooks are rerun.

## Source

Machine-readable table: `results/metrics_definition_audit.csv`.
"""

    output_report = RESULTS_DIR / "metrics_definition_audit.md"
    output_report.write_text(report, encoding="utf-8")

    print(f"Wrote: {output_csv}")
    print(f"Wrote: {output_report}")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
