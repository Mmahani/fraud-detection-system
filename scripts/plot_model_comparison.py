"""Generate the canonical PR-AUC comparison figure from tracked results.

The figure uses Average Precision (``average_precision_score``), the project's
canonical PR-AUC definition. It intentionally does not use the historical
trapezoidal values stored in the Phase 6 notebooks.

Run from the project root:

    python scripts/plot_model_comparison.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "results" / "model_comparison.csv"
FINAL_PATH = ROOT / "results" / "final_threshold_test_results.csv"
OUTPUT_SVG = ROOT / "docs" / "assets" / "model_comparison_pr_auc.svg"
OUTPUT_PNG = ROOT / "docs" / "assets" / "model_comparison_pr_auc.png"

MODELS = [
    ("SGD", "SGD baseline"),
    ("Gradient Boosting", "Gradient Boosting"),
    ("XGBoost", "XGBoost"),
    ("Random Forest", "Random Forest (untuned)"),
    ("Random Forest Tuned", "Random Forest (tuned, final)"),
]


def _load_scores() -> dict[str, float]:
    scores: dict[str, float] = {}

    with COMPARISON_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[row["Model"]] = float(row["PR-AUC"])

    with FINAL_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "Random Forest Tuned":
                scores["Random Forest Tuned"] = float(row["pr_auc"])

    missing = [key for key, _ in MODELS if key not in scores]
    if missing:
        raise ValueError(f"missing model rows in result files: {missing}")

    for key, _ in MODELS:
        if not 0.0 <= scores[key] <= 1.0:
            raise ValueError(f"PR-AUC out of range for {key}: {scores[key]}")

    return scores


def _normalize_svg(path: Path) -> None:
    """Keep the generated SVG deterministic, LF-only, and free of trailing spaces."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Matplotlib uses random identifiers for clip paths and markers. Replacing
    # them with a stable sequence keeps repeated runs byte-for-byte identical.
    identifiers = re.findall(r'id="([pm][0-9a-f]{6,})"', text)
    mapping: dict[str, str] = {}
    counters = {"p": 0, "m": 0}
    for identifier in identifiers:
        if identifier not in mapping:
            prefix = identifier[0]
            mapping[identifier] = f"{prefix}{counters[prefix]}"
            counters[prefix] += 1

    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    lines = [line.rstrip() for line in text.split("\n")]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    scores = _load_scores()
    labels = [label for _, label in MODELS]
    values = [scores[key] for key, _ in MODELS]
    colors = ["#94a3b8"] * (len(MODELS) - 1) + ["#059669"]
    best = max(values)

    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=160)
    bars = ax.barh(labels, values, color=colors, height=0.62)

    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.86)
    ax.set_xlabel("Average Precision (PR-AUC)", fontsize=11, color="#334155")
    ax.set_title(
        "Canonical PR-AUC on the reserved IEEE-CIS test split",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10.5, colors="#0f172a")
    ax.tick_params(axis="x", labelsize=9.5, colors="#475569")

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.6f}",
            va="center",
            fontsize=10,
            color="#0f172a",
            fontweight="bold" if value == best else "normal",
        )

    fig.text(
        0.01,
        0.015,
        "Source: results/model_comparison.csv and results/final_threshold_test_results.csv. "
        "Retrospective test split; PR-AUC is threshold-independent.",
        fontsize=8,
        color="#64748b",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_SVG,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    fig.savefig(OUTPUT_PNG, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _normalize_svg(OUTPUT_SVG)

    print(f"Wrote {OUTPUT_SVG.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_PNG.relative_to(ROOT)}")
    for key, label in MODELS:
        print(f"  {label}: {scores[key]:.6f}")


if __name__ == "__main__":
    main()
