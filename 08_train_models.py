from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CIC-IDS2017 -> SAME DATA PIPELINE AS THE OTHER DATASET
# STAGE 08: TRAIN ISOLATION FOREST + ONE-CLASS SVM
# ============================================================
#
# Input:
#   train_behavior.csv
#
# Output:
#   selected_features.json
#   behavior_feature_selection_summary.csv
#   behavior_correlation_matrix.csv
#   anomaly_model.pkl
#   behavior_isolation_forest.pkl
#   ocsvm_model.pkl
#   behavior_tuning_config.json
#
# Feature selection:
#   - TRAIN ONLY
#   - remove constant features
#   - remove one feature from each pair with |correlation| > 0.90
#
# Model parameters are matched to the other dataset:
#   Isolation Forest:
#       n_estimators=300
#       contamination=0.01
#       random_state=42
#       n_jobs=-1
#   One-Class SVM:
#       kernel='rbf'
#       nu=0.01
#       gamma='scale'
#
# NOTE:
#   max_samples and max_features are intentionally NOT set here,
#   matching the other dataset's train_model.py.
#   Therefore sklearn's defaults are used.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_behavior.csv"
SELECTED_FILE = BASE_DIR / "selected_features.json"
FEATURE_SUMMARY_FILE = BASE_DIR / "behavior_feature_selection_summary.csv"
CORRELATION_FILE = BASE_DIR / "behavior_correlation_matrix.csv"
IF_MODEL_FILE = BASE_DIR / "anomaly_model.pkl"
LEGACY_MODEL_FILE = BASE_DIR / "behavior_isolation_forest.pkl"
OCSVM_MODEL_FILE = BASE_DIR / "ocsvm_model.pkl"
CONFIG_FILE = BASE_DIR / "behavior_tuning_config.json"

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.90
IF_N_ESTIMATORS = 300
IF_CONTAMINATION = 0.01
OCSVM_KERNEL = "rbf"
OCSVM_NU = 0.01
OCSVM_GAMMA = "scale"


# These are the behavior features that can be used if they exist in the
# prepared CIC behavior dataset. Identifier/label columns are excluded below.
CANDIDATE_FEATURES = [
    "total_requests",
    "request_rate",
    "throughput",
    "flow_count",
    "error_rate",
    "count_4xx",
    "count_5xx",
    "avg_latency",
    "max_latency",
    "latency_std_dev",
    "unique_ip",
    "new_ip_ratio",
    "active_sessions",
    "total_in_packets",
    "total_out_packets",
]

ID_AND_LABEL_COLUMNS = {
    "window",
    "window_id",
    "source_file",
    "label",
    "Label",
    "target",
    "behavior_label",
    "attack_type",
    "flow_count",
    "benign_flow_count",
    "attack_flow_count",
    "attack_ratio",
    "window_purity",
}


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")


def numeric_clean(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def select_candidate_features(train_df: pd.DataFrame) -> list[str]:
    # Prefer the known behavior-feature order so downstream files are stable.
    candidates = []
    for feature in CANDIDATE_FEATURES:
        if feature in train_df.columns and feature not in ID_AND_LABEL_COLUMNS:
            numeric = numeric_clean(train_df[feature])
            if numeric.notna().any():
                candidates.append(feature)

    # Fallback: include any additional numeric columns not explicitly excluded.
    for column in train_df.columns:
        if column in ID_AND_LABEL_COLUMNS or column in candidates:
            continue
        numeric = numeric_clean(train_df[column])
        if numeric.notna().any():
            candidates.append(column)

    if not candidates:
        raise ValueError("ไม่พบ numeric behavior features สำหรับ train")

    return candidates


def calculate_feature_selection(train_df: pd.DataFrame, candidate_features: list[str]):
    x = pd.DataFrame(index=train_df.index)
    for feature in candidate_features:
        x[feature] = numeric_clean(train_df[feature])

    # Use column medians only for the purpose of correlation calculation.
    # This does not alter the saved training dataset.
    x_corr = x.copy()
    for feature in x_corr.columns:
        median = x_corr[feature].median()
        if pd.isna(median):
            median = 0.0
        x_corr[feature] = x_corr[feature].fillna(median)

    constant_features = [
        feature
        for feature in candidate_features
        if x_corr[feature].nunique(dropna=False) <= 1
    ]

    non_constant = [
        feature for feature in candidate_features
        if feature not in constant_features
    ]

    corr = x_corr[non_constant].corr().abs()

    if corr.empty:
        features_to_drop = []
    else:
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        features_to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > CORRELATION_THRESHOLD)
        ]

    selected = [
        feature
        for feature in non_constant
        if feature not in features_to_drop
    ]

    if not selected:
        raise ValueError("หลัง Feature Selection ไม่เหลือ feature สำหรับ train")

    return x_corr, constant_features, features_to_drop, selected, corr


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("CIC-IDS2017 BEHAVIOR MODEL TRAINING")
    print("SAME MODEL WORKFLOW AS THE OTHER DATASET")
    print("=" * 80)

    require_file(TRAIN_FILE)
    train_df = pd.read_csv(TRAIN_FILE)

    if "target" in train_df.columns:
        target = pd.to_numeric(train_df["target"], errors="coerce").fillna(0).astype(int)
        if (target != 0).any():
            raise ValueError("train_behavior.csv ต้องเป็น BENIGN/Normal เท่านั้น")

    candidate_features = select_candidate_features(train_df)
    print(f"Train rows              : {len(train_df):,}")
    print(f"Candidate features      : {len(candidate_features)}")

    x_corr, constant_features, features_to_drop, selected_features, corr = (
        calculate_feature_selection(train_df, candidate_features)
    )

    print()
    print("=" * 80)
    print("FEATURE SELECTION - TRAIN ONLY")
    print("=" * 80)
    print(f"Constant removed        : {len(constant_features)}")
    print(f"Correlation removed     : {len(features_to_drop)}")
    print(f"Final model features    : {len(selected_features)}")
    print()
    for index, feature in enumerate(selected_features, start=1):
        print(f"{index:02d}. {feature}")

    corr.to_csv(CORRELATION_FILE)

    summary_rows = []
    for feature in candidate_features:
        summary_rows.append(
            {
                "feature": feature,
                "constant_on_train": feature in constant_features,
                "high_correlation_on_train": feature in features_to_drop,
                "selected_for_model": feature in selected_features,
            }
        )
    pd.DataFrame(summary_rows).to_csv(FEATURE_SUMMARY_FILE, index=False)

    with open(SELECTED_FILE, "w", encoding="utf-8") as file:
        json.dump(selected_features, file, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Build X exactly from the selected TRAIN-only features.
    # --------------------------------------------------------
    x_train = train_df[selected_features].copy()
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    x_train = x_train.to_numpy(dtype=np.float64)

    # --------------------------------------------------------
    # Isolation Forest — same parameters as the other dataset
    # --------------------------------------------------------
    print()
    print("=" * 80)
    print("TRAINING ISOLATION FOREST")
    print("=" * 80)
    print(f"n_estimators  : {IF_N_ESTIMATORS}")
    print(f"contamination : {IF_CONTAMINATION}")
    print("random_state  : 42")
    print("n_jobs        : -1")

    iso_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "iso",
            IsolationForest(
                n_estimators=IF_N_ESTIMATORS,
                contamination=IF_CONTAMINATION,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])
    iso_model.fit(x_train)
    joblib.dump(iso_model, IF_MODEL_FILE)
    joblib.dump(iso_model, LEGACY_MODEL_FILE)
    print(f"Saved: {IF_MODEL_FILE.name}")
    print(f"Saved: {LEGACY_MODEL_FILE.name}")

    # --------------------------------------------------------
    # One-Class SVM — same parameters as the other dataset
    # --------------------------------------------------------
    print("=" * 80)
    print("TRAINING ONE-CLASS SVM")
    print("=" * 80)
    print(f"kernel : {OCSVM_KERNEL}")
    print(f"nu     : {OCSVM_NU}")
    print(f"gamma  : {OCSVM_GAMMA}")

    ocsvm_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "ocsvm",
            OneClassSVM(
                kernel=OCSVM_KERNEL,
                nu=OCSVM_NU,
                gamma=OCSVM_GAMMA,
            ),
        ),
    ])
    ocsvm_model.fit(x_train)
    joblib.dump(ocsvm_model, OCSVM_MODEL_FILE)
    print(f"Saved: {OCSVM_MODEL_FILE.name}")

    config = {
        "workflow": "same_as_other_dataset",
        "random_state": RANDOM_STATE,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "selected_features": selected_features,
        "isolation_forest": {
            "n_estimators": IF_N_ESTIMATORS,
            "contamination": IF_CONTAMINATION,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "one_class_svm": {
            "kernel": OCSVM_KERNEL,
            "nu": OCSVM_NU,
            "gamma": OCSVM_GAMMA,
        },
        "evaluation_threshold": 0.70,
        "attack_ratio_to_normal_test": 0.50,
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print()
    print("Selected features saved : selected_features.json")
    print("Config saved             : behavior_tuning_config.json")
    print("=" * 80)
    print("STAGE 08 SAME-WORKFLOW TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
