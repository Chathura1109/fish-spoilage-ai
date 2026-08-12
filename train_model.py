"""Train and evaluate the fish spoilage decision-support model.

The current CSV is synthetic prototype data. This script validates its physical
constraints, keeps batches out of both train and test sets, calibrates predicted
probabilities, and stores the preprocessing and model as one deployable bundle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# File locations and model-wide settings
# ---------------------------------------------------------------------------
# Resolve files from this script's folder so commands work from any directory.
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "fish_spoilage_dataset.csv"
MODEL_PATH = BASE_DIR / "spoilage_model.joblib"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
METRICS_PATH = BASE_DIR / "model_metrics.json"

MODEL_VERSION = "spoilage-v3"
RANDOM_STATE = 42
TARGET_COLUMN = "spoilage_risk"
BATCH_COLUMN = "batch_id"
VALID_LABELS = {"LOW", "MEDIUM", "HIGH"}

CATEGORICAL_FEATURES = ["fish_species"]
NUMERIC_FEATURES = [
    "has_temperature_telemetry",
    "temperature_reading_count",
    "current_temperature",
    "average_temperature",
    "minimum_temperature",
    "maximum_temperature",
    "air_temperature",
    "storage_duration_hours",
    "humidity",
    "transport_duration_hours",
    "time_above_limit_minutes",
    "temperature_violation_count",
    "time_since_catch_hours",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Broad engineering limits for the prototype API. These are input-integrity
# limits, not official food-safety thresholds.
INPUT_BOUNDS = {
    "has_temperature_telemetry": (0, 1),
    "temperature_reading_count": (0, 1_000_000),
    "current_temperature": (-5.0, 40.0),
    "average_temperature": (-5.0, 40.0),
    "minimum_temperature": (-10.0, 40.0),
    "maximum_temperature": (-5.0, 50.0),
    "air_temperature": (-10.0, 50.0),
    "storage_duration_hours": (0.0, 8760.0),
    "humidity": (0.0, 100.0),
    "transport_duration_hours": (0.0, 720.0),
    "time_above_limit_minutes": (0.0, 525_600.0),
    "temperature_violation_count": (0, 10_000),
    "time_since_catch_hours": (0.0, 8760.0),
}


def load_and_validate_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return training-ready data and an audit of every cleaning decision.

    Contradictory rows are rejected because teaching a model from impossible
    synthetic examples makes its predictions less useful. Missing feature
    values are allowed because the preprocessing pipeline imputes them later.
    """
    # Fail early with a clear message when the CSV schema is incomplete.
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Normalise text and convert numeric-looking strings into numbers. Failed
    # conversions become NaN and are handled by the imputers during training.
    audit: dict[str, object] = {"rows_loaded": int(len(df))}
    df = df.copy()
    df["fish_species"] = df["fish_species"].astype("string").str.strip().str.title()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("string").str.strip().str.upper()
    telemetry_values = (
        df["has_temperature_telemetry"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    )
    df["has_temperature_telemetry"] = telemetry_values
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # `invalid` combines all rejection rules. `reasons` keeps separate counts
    # so model_metrics.json explains exactly why rows were excluded.
    invalid = pd.Series(False, index=df.index)
    reasons: dict[str, int] = {}

    def reject(name: str, condition: pd.Series) -> None:
        nonlocal invalid
        condition = condition.fillna(False)
        reasons[name] = int(condition.sum())
        invalid |= condition

    reject("missing_or_unknown_target", ~df[TARGET_COLUMN].isin(VALID_LABELS))
    reject("missing_species", df["fish_species"].isna() | df["fish_species"].eq(""))
    reject(
        "missing_telemetry_availability",
        df["has_temperature_telemetry"].isna(),
    )
    reject(
        "missing_temperature_reading_count",
        df["temperature_reading_count"].isna(),
    )

    # Check each numeric input against broad prototype engineering limits.
    for column, (minimum, maximum) in INPUT_BOUNDS.items():
        value = df[column]
        reject(
            f"{column}_outside_bounds",
            value.notna() & ~value.between(minimum, maximum),
        )

    for column in ["temperature_reading_count", "temperature_violation_count"]:
        value = df[column]
        reject(f"{column}_not_integer", value.notna() & value.mod(1).ne(0))
    reject(
        "current_temperature_above_maximum",
        df["current_temperature"].notna()
        & df["maximum_temperature"].notna()
        & df["current_temperature"].gt(df["maximum_temperature"]),
    )
    reject(
        "average_temperature_above_maximum",
        df["average_temperature"].notna()
        & df["maximum_temperature"].notna()
        & df["average_temperature"].gt(df["maximum_temperature"]),
    )
    reject(
        "minimum_temperature_above_current",
        df["minimum_temperature"].notna()
        & df["current_temperature"].notna()
        & df["minimum_temperature"].gt(df["current_temperature"]),
    )
    reject(
        "minimum_temperature_above_average",
        df["minimum_temperature"].notna()
        & df["average_temperature"].notna()
        & df["minimum_temperature"].gt(df["average_temperature"]),
    )
    reject(
        "transport_duration_above_storage_duration",
        df["transport_duration_hours"].notna()
        & df["storage_duration_hours"].notna()
        & df["transport_duration_hours"].gt(df["storage_duration_hours"]),
    )
    reject(
        "storage_duration_above_time_since_catch",
        df["storage_duration_hours"].notna()
        & df["time_since_catch_hours"].notna()
        & df["storage_duration_hours"].gt(df["time_since_catch_hours"]),
    )
    reject(
        "time_above_limit_exceeds_storage_duration",
        df["time_above_limit_minutes"].notna()
        & df["storage_duration_hours"].notna()
        & df["time_above_limit_minutes"].gt(df["storage_duration_hours"] * 60),
    )

    product_temperature_columns = [
        "current_temperature",
        "average_temperature",
        "minimum_temperature",
        "maximum_temperature",
    ]
    has_telemetry = df["has_temperature_telemetry"].eq(1)
    lacks_telemetry = df["has_temperature_telemetry"].eq(0)
    reject(
        "telemetry_present_without_readings",
        has_telemetry
        & (
            df["temperature_reading_count"].fillna(0).le(0)
            | df[product_temperature_columns].isna().any(axis=1)
        ),
    )
    reject(
        "telemetry_absent_with_product_measurements",
        lacks_telemetry
        & (
            df["temperature_reading_count"].fillna(0).ne(0)
            | df[product_temperature_columns].notna().any(axis=1)
            | df["time_above_limit_minutes"].fillna(0).ne(0)
            | df["temperature_violation_count"].fillna(0).ne(0)
        ),
    )
    reject(
        "violations_exceed_reading_count",
        df["temperature_violation_count"].notna()
        & df["temperature_reading_count"].notna()
        & df["temperature_violation_count"].gt(df["temperature_reading_count"]),
    )

    audit["invalid_row_reasons"] = reasons
    audit["rows_rejected"] = int(invalid.sum())
    df = df.loc[~invalid].copy()

    # Exact duplicates could make held-out evaluation look better than it is.
    before_deduplication = len(df)
    df = df.drop_duplicates(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    audit["duplicate_rows_removed"] = int(before_deduplication - len(df))
    audit["missing_feature_values_to_impute"] = {
        column: int(df[column].isna().sum()) for column in FEATURE_COLUMNS
    }

    if BATCH_COLUMN not in df.columns:
        # The synthetic file has one sample per row. Treating each row as a
        # distinct batch makes the group boundary explicit without inventing
        # repeated measurements that do not exist.
        df[BATCH_COLUMN] = [f"SYNTH-BATCH-{index:06d}" for index in range(len(df))]
        audit["batch_id_source"] = "generated_unique_id_per_synthetic_sample"
    else:
        df[BATCH_COLUMN] = df[BATCH_COLUMN].astype("string").str.strip()
        missing_batch = df[BATCH_COLUMN].isna() | df[BATCH_COLUMN].eq("")
        if missing_batch.any():
            raise ValueError("batch_id contains missing values")
        audit["batch_id_source"] = "dataset"

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} valid rows remain; at least 100 are required")
    audit["rows_used"] = int(len(df))
    audit["unique_batches"] = int(df[BATCH_COLUMN].nunique())
    return df, audit


def main() -> None:
    """Train, evaluate, explain, and save a complete model bundle."""
    # STEP 1: Load and clean the synthetic CSV.
    df, data_audit = load_and_validate_dataset(DATASET_PATH)
    print(json.dumps(data_audit, indent=2))

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df[BATCH_COLUMN]

    # STEP 2: Hold out one group-aware fold for final testing. With real data,
    # this prevents readings from the same batch appearing on both sides.
    outer_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_index, test_index = next(outer_splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    groups_train = groups.iloc[train_index]
    groups_test = groups.iloc[test_index]

    # Turn accidental batch leakage into an immediate training failure.
    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise RuntimeError(f"Batch leakage detected across split: {sorted(overlap)[:5]}")

    # STEP 3: Create group-safe calibration folds from training data only.
    # Calibration makes the returned probabilities more interpretable.
    calibration_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1
    )
    calibration_folds = list(calibration_splitter.split(X_train, y_train, groups_train))

    # STEP 4: Species is a name, not an ordered number, so it is one-hot
    # encoded. Missing numeric inputs are replaced with the training median.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("species", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
    # Keep preprocessing and Random Forest together so training and API
    # preprocessing cannot accidentally drift apart.
    base_model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    # Missing HIGH risk is the most important error, so that
                    # class receives twice the training weight.
                    class_weight={"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 2.0},
                ),
            ),
        ]
    )
    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=True,
    )
    # STEP 5: Fit the calibrated model using only training folds.
    model.fit(X_train, y_train)

    # STEP 6: Evaluate once on the untouched group-held-out test fold.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    labels = ["LOW", "MEDIUM", "HIGH"]
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_is_synthetic": True,
        "decision_support_only": True,
        "data_audit": data_audit,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_batches": int(groups_train.nunique()),
        "test_batches": int(groups_test.nunique()),
        "batch_overlap_count": 0,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "multiclass_log_loss": float(log_loss(y_test, probabilities, labels=model.classes_)),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
    }

    # Permutation importance measures the macro-F1 drop after shuffling one
    # feature. It is easier to explain than internal tree importance here.
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    metrics["permutation_importance_f1_macro"] = {
        feature: float(importance)
        for feature, importance in sorted(
            zip(FEATURE_COLUMNS, permutation.importances_mean, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    print("\n=== Classification Report ===")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"HIGH recall: {report['HIGH']['recall']:.4f}")
    print(f"Batch overlap: {metrics['batch_overlap_count']}")

    # STEP 7: Save a confusion matrix suitable for the university report.
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues")
    plt.title("Fish Spoilage Risk - Group-Held-Out Test Set")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    # STEP 8: Save everything FastAPI needs as one versioned artifact.
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "known_species": sorted(df["fish_species"].dropna().unique().tolist()),
        "input_bounds": INPUT_BOUNDS,
        "model_version": MODEL_VERSION,
        "trained_at_utc": metrics["trained_at_utc"],
        "dataset_is_synthetic": True,
        "decision_support_only": True,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH.name}, {METRICS_PATH.name}, and {CONFUSION_MATRIX_PATH.name}")


if __name__ == "__main__":
    main()
