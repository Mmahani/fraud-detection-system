"""Build and validate the project artifact manifest.

The manifest records the files required to run inference, reproduce the
reported evaluation, and complete the explainability deliverables. Optional
artifacts (for example, processed data and Phase 10 CSV files) are recorded but
do not block inference validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifest.json"

ARTIFACT_SPECS: list[dict[str, Any]] = [
    # Inference-critical artifacts.
    {
        "path": "models/best_tuned_model.pkl",
        "group": "inference",
        "required": True,
        "description": "Final tuned RandomForestClassifier",
    },
    {
        "path": "models/preprocessor.pkl",
        "group": "inference",
        "required": True,
        "description": "Fitted preprocessing pipeline used by inference",
    },
    {
        "path": "models/threshold_config.json",
        "group": "inference",
        "required": True,
        "description": "Frozen classification threshold and risk bands",
    },
    {
        "path": "models/model_metadata.json",
        "group": "inference",
        "required": True,
        "description": "Model metadata used by packaging and reporting",
    },
    {
        "path": "models/explainability_config.json",
        "group": "inference",
        "required": True,
        "description": "SHAP explainability configuration",
    },
    # Evaluation and reporting artifacts.
    {
        "path": "results/final_threshold_test_results.csv",
        "group": "evaluation",
        "required": True,
        "description": "Frozen-threshold test metrics",
    },
    {
        "path": "results/phase8_fast_summary.json",
        "group": "evaluation",
        "required": True,
        "description": "Phase 8 CV and test summary",
    },
    {
        "path": "results/phase8_evaluation_audit.md",
        "group": "evaluation",
        "required": True,
        "description": "Phase 8 metric and leakage audit",
    },
    {
        "path": "results/phase9_final_report.md",
        "group": "evaluation",
        "required": True,
        "description": "Phase 9 threshold report",
    },
    {
        "path": "results/best_model_parameters.json",
        "group": "evaluation",
        "required": True,
        "description": "Selected model parameters and search metadata",
    },
    {
        "path": "results/final_model_comparison_phase8.csv",
        "group": "evaluation",
        "required": True,
        "description": "Final model comparison table",
    },
    {
        "path": "results/phase8_fast_report.md",
        "group": "evaluation",
        "required": True,
        "description": "Phase 8 narrative report",
    },
    {
        "path": "results/phase8_tuned_vs_untuned_test.csv",
        "group": "evaluation",
        "required": True,
        "description": "Tuned versus untuned test comparison",
    },
    {
        "path": "results/model_comparison.csv",
        "group": "evaluation",
        "required": True,
        "description": "Phase 6 model comparison table",
    },
    {
        "path": "results/hyperparameter_search_results.csv",
        "group": "evaluation",
        "required": True,
        "description": "Phase 8 hyperparameter search results",
    },
    {
        "path": "results/threshold_analysis.csv",
        "group": "evaluation",
        "required": True,
        "description": "OOF threshold analysis",
    },
    {
        "path": "results/threshold_constraints_oof.csv",
        "group": "evaluation",
        "required": True,
        "description": "OOF threshold constraint checks",
    },
    {
        "path": "results/threshold_050_vs_final_test.csv",
        "group": "evaluation",
        "required": True,
        "description": "Test comparison at thresholds 0.50 and 0.35",
    },
    {
        "path": "results/threshold_sensitivity_test.csv",
        "group": "evaluation",
        "required": True,
        "description": "Threshold sensitivity analysis",
    },
    {
        "path": "results/final_prediction_examples.csv",
        "group": "evaluation",
        "required": True,
        "description": "Phase 11 prediction examples",
    },
    {
        "path": "results/metrics_definition_audit.md",
        "group": "evaluation",
        "required": True,
        "description": "Audit of canonical versus historical PR-AUC definitions",
    },
    {
        "path": "results/metrics_definition_audit.csv",
        "group": "evaluation",
        "required": True,
        "description": "Machine-readable PR-AUC audit table",
    },
    {
        "path": "results/phase12_time_validation.csv",
        "group": "evaluation",
        "required": True,
        "description": "Phase 12 time-bucket diagnostic table",
    },
    {
        "path": "results/phase12_time_validation.md",
        "group": "evaluation",
        "required": True,
        "description": "Phase 12 time-based validation report",
    },
    {
        "path": "results/phase12_summary.json",
        "group": "evaluation",
        "required": True,
        "description": "Phase 12 diagnostic summary",
    },
    {
        "path": "results/phase12_plots/time_bucket_pr_auc.png",
        "group": "evaluation",
        "required": False,
        "description": "Phase 12 time-bucket PR-AUC plot",
    },
    {
        "path": "results/improvement_comparison.csv",
        "group": "evaluation",
        "required": False,
        "description": "Archived Phase 8 improvement experiment; not the final model",
    },
    {
        "path": "results/oof_proba.npy",
        "group": "evaluation",
        "required": False,
        "description": "OOF probabilities used for threshold selection; regenerate with Phase 9",
    },
    # Explainability outputs. The two plots exist; the CSV/report files are
    # still missing and are recorded as optional so validation can flag them.
    {
        "path": "results/phase10_plots/shap_beeswarm.png",
        "group": "explainability",
        "required": False,
        "description": "SHAP beeswarm plot",
    },
    {
        "path": "results/phase10_plots/shap_global_bar.png",
        "group": "explainability",
        "required": False,
        "description": "SHAP global bar plot",
    },
    {
        "path": "results/shap_global_importance.csv",
        "group": "explainability",
        "required": False,
        "description": "Global SHAP importance table (missing; regenerate Phase 10)",
    },
    {
        "path": "results/shap_global_importance_top20.csv",
        "group": "explainability",
        "required": False,
        "description": "Top-20 SHAP importance table (missing; regenerate Phase 10)",
    },
    {
        "path": "results/shap_fraud_direction.csv",
        "group": "explainability",
        "required": False,
        "description": "SHAP direction summary (missing; regenerate Phase 10)",
    },
    {
        "path": "results/shap_local_explanations.csv",
        "group": "explainability",
        "required": False,
        "description": "Local SHAP explanation examples (missing; regenerate Phase 10)",
    },
    {
        "path": "results/phase10_final_report.md",
        "group": "explainability",
        "required": False,
        "description": "Phase 10 report (missing; regenerate Phase 10)",
    },
    # Processed data. These files are intentionally excluded from Git, so a
    # fresh clone will report them as optional missing artifacts.
    {
        "path": "data/processed/merged_data.csv.gz",
        "group": "data",
        "required": False,
        "description": "Merged raw transaction and identity data",
    },
    {
        "path": "data/processed/X_train_processed.npz",
        "group": "data",
        "required": False,
        "description": "Processed training feature matrix",
    },
    {
        "path": "data/processed/X_test_processed.npz",
        "group": "data",
        "required": False,
        "description": "Processed test feature matrix",
    },
    {
        "path": "data/processed/y_train.csv",
        "group": "data",
        "required": False,
        "description": "Training labels",
    },
    {
        "path": "data/processed/y_test.csv",
        "group": "data",
        "required": False,
        "description": "Test labels",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _manifest_environment() -> dict[str, Any]:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]:
        packages[name] = _package_version(name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def build_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    strict: bool = True,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Build the artifact manifest and write it to ``manifest_path``."""
    files: list[dict[str, Any]] = []
    missing_required: list[str] = []

    for spec in ARTIFACT_SPECS:
        path = PROJECT_ROOT / spec["path"]
        entry: dict[str, Any] = {
            "path": spec["path"],
            "group": spec["group"],
            "required": bool(spec["required"]),
            "description": spec["description"],
        }
        if path.is_file():
            entry["status"] = "present"
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = _sha256(path) if hash_files else None
        else:
            entry["status"] = "missing"
            entry["size_bytes"] = None
            entry["sha256"] = None
            if spec["required"]:
                missing_required.append(spec["path"])
        files.append(entry)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": "fraud-detection-system",
        "generated_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "files": files,
        "summary": {
            "total": len(files),
            "present": sum(1 for item in files if item["status"] == "present"),
            "missing": sum(1 for item in files if item["status"] == "missing"),
            "missing_required": missing_required,
        },
        "environment": _manifest_environment(),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if strict and missing_required:
        raise FileNotFoundError(
            "Missing required artifacts: " + ", ".join(missing_required)
        )
    return manifest


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    check_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a manifest against the files currently on disk."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    required_checked = 0
    optional_checked = 0

    for entry in manifest.get("files", []):
        checked += 1
        required = bool(entry.get("required", False))
        if required:
            required_checked += 1
        else:
            optional_checked += 1

        relative_path = entry.get("path")
        if not relative_path:
            errors.append("Manifest entry is missing a path")
            continue

        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            message = f"Missing {'required' if required else 'optional'} artifact: {relative_path}"
            if required:
                errors.append(message)
            else:
                warnings.append(message)
            continue

        expected_hash = entry.get("sha256")
        if check_hashes and expected_hash:
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                message = f"Hash mismatch: {relative_path}"
                if required:
                    errors.append(message)
                else:
                    warnings.append(message)

    return {
        "valid": not errors,
        "checked": checked,
        "required_checked": required_checked,
        "optional_checked": optional_checked,
        "errors": errors,
        "warnings": warnings,
        "manifest": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the fraud-detection artifact manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build artifacts/manifest.json")
    build_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest output path",
    )
    build_parser.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Write the manifest even if required artifacts are missing",
    )
    build_parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Do not compute SHA-256 hashes",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate the artifact manifest"
    )
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest path to validate",
    )
    validate_parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Check presence only, not SHA-256 hashes",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        try:
            manifest = build_manifest(
                args.manifest,
                strict=not args.allow_missing_required,
                hash_files=not args.skip_hashes,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Manifest written: {args.manifest}")
        print(json.dumps(manifest["summary"], indent=2))
        return 0

    if args.command == "validate":
        try:
            result = validate_manifest(
                args.manifest,
                check_hashes=not args.skip_hashes,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
