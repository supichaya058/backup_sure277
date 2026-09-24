# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ============================================================
# CIC-IDS2017 -> ONE-CLASS SVM THRESHOLD TUNING
# ============================================================
#
# Purpose:
#   Tune the One-Class SVM anomaly-detection model using the SAME
#   Behavior Dataset and the SAME selected features used by the
#   Isolation Forest pipeline.
#
# Input:
#   train_behavior.csv
#   validation_behavior.csv
#   selected_behavior_features.csv
#   behavior_feature_selection_summary.csv (fallback only)
#
# Training:
#   BENIGN Behavior only
#
# Validation:
#   BENIGN + ATTACK
#
# IMPORTANT:
#   Final Test is NOT used in this stage.
#   Stage 17 tunes ONLY the anomaly-score detection threshold.
#   One-Class SVM hyperparameters are fixed to the Stage 16 baseline.
#   Hyperparameter tuning is performed separately in Stage 18.
#
# Score convention:
#   OneClassSVM decision_function():
#       high  = more normal
#       low   = more anomalous
#
#   Therefore:
#       anomaly_score = -decision_function()
#
#   Higher anomaly_score = more anomalous.
#
# Tuning dimensions:
#   nu
#   gamma
#   anomaly-score threshold
#
# IMPORTANT THRESHOLD NOTE:
#   One-Class SVM has a native decision boundary at
#   decision_function() = 0. The previous version of Stage 17
#   only tested 90-99th percentile thresholds, which forced the
#   model to label only the most extreme 1-10% of validation
#   samples as anomalies. That was too restrictive for this
#   dataset and could not preserve the Stage 16 native-boundary
#   result.
#
#   The fixed version therefore includes the native threshold 0
#   and a wider percentile range for Validation tuning.
#
# Selection rule (same research protocol):
#   1. Keep only configurations with Validation Normal-only FAR <= 15%.
#   2. Among eligible configurations, maximize Validation F1 Score.
#
# Tie-breakers:
#   1. Higher Validation Recall
#   2. Lower Validation False Positive Rate
#   3. Higher Validation Precision
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = (
    BASE_DIR / "train_behavior.csv"
)

VALIDATION_FILE = (
    BASE_DIR / "validation_behavior.csv"
)

FEATURE_FILE = (
    BASE_DIR / "selected_behavior_features.csv"
)

FEATURE_SELECTION_SUMMARY_FILE = (
    BASE_DIR / "behavior_feature_selection_summary.csv"
)

BEST_MODEL_FILE = (
    BASE_DIR / "one_class_svm_threshold_baseline_validation.pkl"
)

RESULT_FILE = (
    BASE_DIR / "one_class_svm_threshold_tuning_results.csv"
)

BEST_RESULT_FILE = (
    BASE_DIR / "one_class_svm_threshold_config.json"
)

SUMMARY_FILE = (
    BASE_DIR / "one_class_svm_threshold_summary.txt"
)

VALIDATION_SCORE_FILE = (
    BASE_DIR / "one_class_svm_threshold_validation_scores.csv"
)

RANDOM_STATE = 42

# ------------------------------------------------------------
# Stage 16 baseline parameters (FIXED in Stage 17)
# ------------------------------------------------------------
# Stage 17 follows the Isolation Forest sequence:
#   Stage 16 = baseline model
#   Stage 17 = threshold tuning only
#   Stage 18 = hyperparameter tuning
#
OCSVM_BASELINE_NU = 0.01
OCSVM_BASELINE_GAMMA = "scale"

# ------------------------------------------------------------
# One-Class SVM hyperparameter search space for Stage 18
# ------------------------------------------------------------
# ------------------------------------------------------------
#
# The search is deliberately moderate because RBF One-Class SVM
# can be computationally expensive on large training sets.
#
# nu controls the allowed fraction of training observations
# considered as outliers / support-vector behavior.
#
# gamma controls the influence radius of the RBF kernel.
#
NU_VALUES = [
    0.001,
    0.005,
    0.01,
    0.03,
    0.05,
    0.10,
]

GAMMA_VALUES = [
    "scale",
    0.001,
    0.01,
    0.1,
    1.0,
]

# Tune the anomaly-score decision threshold on Validation.
#
# 0 is the native One-Class SVM boundary because
# decision_function() >= 0 means the sample is classified as
# normal by the native OCSVM decision rule, and therefore
# anomaly_score = -decision_function() >= 0 marks an anomaly.
#
# Percentile thresholds are included across a much wider range
# so the search is not forced to label only the top 1-10% of
# validation scores as anomalies.
THRESHOLD_PERCENTILES = [
    20,
    30,
    40,
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    92,
    95,
    97,
    99,
]

KERNEL = "rbf"

CACHE_SIZE = 4096

SHRINKING = True

# Set to None to use ALL BENIGN Train rows.
#
# The baseline Stage 16 successfully trained on the full
# 22,141-row BENIGN training set, so Stage 17 keeps the same
# default for methodological consistency.
#
# If computational cost becomes excessive, a deterministic
# sample size can be supplied and documented as a computational
# experiment.
TRAIN_SAMPLE_LIMIT = None


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"ไม่พบไฟล์: {file_path}"
        )


# ------------------------------------------------------------
# FEATURE FILE RECOVERY
# ------------------------------------------------------------
#
# Stage 08 is the only stage responsible for actual model
# feature selection.
#
# Normally:
#   selected_behavior_features.csv
#
# If the selected feature file is missing, this stage can rebuild
# it from the TRAIN-only Stage 08 selection summary.
#
# This recovery does NOT recalculate feature selection using
# Validation or Final Test.
# ------------------------------------------------------------

if not FEATURE_FILE.exists():

    if not FEATURE_SELECTION_SUMMARY_FILE.exists():

        raise FileNotFoundError(
            "ไม่พบไฟล์สำหรับ Model Features:\n"
            f"- {FEATURE_FILE}\n"
            f"- {FEATURE_SELECTION_SUMMARY_FILE}\n"
            "กรุณารัน Stage 08 ก่อน"
        )

    feature_selection_df = pd.read_csv(
        FEATURE_SELECTION_SUMMARY_FILE
    )

    if "feature" not in feature_selection_df.columns:

        raise ValueError(
            "behavior_feature_selection_summary.csv "
            "ต้องมี column ชื่อ 'feature'"
        )

    if "selected" not in feature_selection_df.columns:

        raise ValueError(
            "behavior_feature_selection_summary.csv "
            "ต้องมี column ชื่อ 'selected'"
        )

    selected_flag = (
        feature_selection_df["selected"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )

    recovered_features = (
        feature_selection_df.loc[
            selected_flag,
            "feature",
        ]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    if not recovered_features:

        raise ValueError(
            "ไม่พบ Features ที่ถูกเลือกใน "
            "behavior_feature_selection_summary.csv"
        )

    pd.DataFrame(
        {
            "feature": recovered_features
        }
    ).to_csv(
        FEATURE_FILE,
        index=False,
    )

    print(
        "Reconstructed selected_behavior_features.csv "
        "from TRAIN-only Stage 08 selection summary."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("CIC-IDS2017 ONE-CLASS SVM BEHAVIOR MODEL")
print("=" * 80)

# ============================================================
# VALIDATION / FINAL TEST INTEGRITY CHECK
# ============================================================
# Validation MUST remain the Stage 07 Validation split.
# Final Test is reserved for final evaluation and must never be
# copied into validation_behavior.csv before this stage runs.
# ============================================================

TEST_DATASET_FILE = BASE_DIR / "test_dataset.csv"

def validate_split_integrity(train_df, validation_df):
    if "target" in train_df.columns:
        if (pd.to_numeric(train_df["target"], errors="coerce").fillna(0).astype(int) != 0).any():
            raise ValueError("Train ต้องมีเฉพาะ BENIGN")

    if len(validation_df) == 0:
        raise ValueError("Validation dataset ว่าง")

    if "target" not in validation_df.columns:
        raise ValueError("Validation dataset ต้องมี column 'target'")

    # Guard against the previous leakage patch: Validation must not be
    # identical to Final Test.
    if TEST_DATASET_FILE.exists():
        test_df = pd.read_csv(TEST_DATASET_FILE)
        if len(test_df) == len(validation_df):
            try:
                if validation_df.reset_index(drop=True).equals(test_df.reset_index(drop=True)):
                    raise ValueError(
                        "พบ DATA LEAKAGE: validation_behavior.csv เหมือน "
                        "test_dataset.csv ซึ่งเป็น Final Test"
                    )
            except ValueError:
                raise
            except Exception:
                pass

    print(f"Validation integrity check : PASS ({len(validation_df):,} rows)")

train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

validate_split_integrity(train_df, validation_df)

features_df = pd.read_csv(
    FEATURE_FILE
)


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

if "feature" not in features_df.columns:

    raise ValueError(
        "selected_behavior_features.csv "
        "ต้องมี column ชื่อ 'feature'"
    )

FEATURES = (
    features_df["feature"]
    .astype(str)
    .str.strip()
    .tolist()
)

FEATURES = list(
    dict.fromkeys(
        FEATURES
    )
)

# flow_count is metadata only.
if "flow_count" in FEATURES:

    FEATURES.remove(
        "flow_count"
    )


# ============================================================
# CHECK DUPLICATE COLUMNS
# ============================================================

for name, df in [
    ("train_behavior.csv", train_df),
    ("validation_behavior.csv", validation_df),
]:

    if df.columns.duplicated().any():

        duplicates = (
            df.columns[
                df.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            f"{name} มี Columns ซ้ำ:\n"
            + "\n".join(
                duplicates
            )
        )


# ============================================================
# CHECK REQUIRED TARGET / METADATA
# ============================================================

for name, df in [
    ("train_behavior.csv", train_df),
    ("validation_behavior.csv", validation_df),
]:

    if "target" not in df.columns:

        raise ValueError(
            f"{name} ไม่มี column 'target'"
        )


# ============================================================
# CHECK SELECTED FEATURES
# ============================================================

for name, df in [
    ("train_behavior.csv", train_df),
    ("validation_behavior.csv", validation_df),
]:

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"{name} ไม่มี Behavior Features:\n"
            + "\n".join(
                missing_features
            )
        )


# ============================================================
# PREPARE TARGET
# ============================================================

train_target = (
    pd.to_numeric(
        train_df["target"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

validation_target = (
    pd.to_numeric(
        validation_df["target"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)


# Training must contain BENIGN only.
train_attack_count = (
    train_target == 1
).sum()

if train_attack_count != 0:

    raise ValueError(
        "Train dataset ต้องมีเฉพาะ BENIGN Behavior "
        f"แต่พบ ATTACK {train_attack_count:,} rows"
    )


# Validation must contain both classes for meaningful evaluation.
validation_benign_count = (
    validation_target == 0
).sum()

validation_attack_count = (
    validation_target == 1
).sum()

if validation_benign_count == 0:

    raise ValueError(
        "Validation dataset ไม่มี BENIGN rows"
    )

if validation_attack_count == 0:

    raise ValueError(
        "Validation dataset ไม่มี ATTACK rows"
    )


# ============================================================
# OPTIONAL TRAIN SAMPLING
# ============================================================

if TRAIN_SAMPLE_LIMIT is not None:

    if TRAIN_SAMPLE_LIMIT <= 0:

        raise ValueError(
            "TRAIN_SAMPLE_LIMIT ต้องมากกว่า 0 หรือใช้ None"
        )

    if TRAIN_SAMPLE_LIMIT < len(train_df):

        train_df = (
            train_df
            .sample(
                n=TRAIN_SAMPLE_LIMIT,
                random_state=RANDOM_STATE,
            )
            .reset_index(drop=True)
        )

        train_target = (
            pd.to_numeric(
                train_df["target"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train_df = (
    train_df[FEATURES]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)

X_validation_df = (
    validation_df[FEATURES]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


# ============================================================
# CHECK NaN / INF
# ============================================================

if X_train_df.isna().any().any():

    bad_columns = (
        X_train_df.columns[
            X_train_df.isna().any()
        ]
        .tolist()
    )

    raise ValueError(
        "Train มี NaN/Inf ใน Features:\n"
        + "\n".join(
            bad_columns
        )
    )

if X_validation_df.isna().any().any():

    bad_columns = (
        X_validation_df.columns[
            X_validation_df.isna().any()
        ]
        .tolist()
    )

    raise ValueError(
        "Validation มี NaN/Inf ใน Features:\n"
        + "\n".join(
            bad_columns
        )
    )


X_train = X_train_df.to_numpy(
    dtype=np.float64
)

X_validation = X_validation_df.to_numpy(
    dtype=np.float64
)

y_validation = validation_target.to_numpy()


# ============================================================
# PRINT DATA SUMMARY
# ============================================================

print()
print(
    f"Train rows       : {len(train_df):,}"
)

print(
    f"Validation rows  : {len(validation_df):,}"
)

print(
    f"Features         : {len(FEATURES)}"
)

print(
    f"Train BENIGN     : "
    f"{(train_target == 0).sum():,}"
)

print(
    f"Train ATTACK     : "
    f"{(train_target == 1).sum():,}"
)

print(
    f"Validation BENIGN: "
    f"{validation_benign_count:,}"
)

print(
    f"Validation ATTACK: "
    f"{validation_attack_count:,}"
)

print()


# ============================================================
# TUNING SEARCH SPACE
# ============================================================

print("=" * 80)
print("ONE-CLASS SVM HYPERPARAMETER / THRESHOLD TUNING")
print("=" * 80)

print(
    f"Kernel                 : {KERNEL}"
)

print(
    f"nu candidates          : {NU_VALUES}"
)

print(
    f"gamma candidates       : {GAMMA_VALUES}"
)

print(
    f"Threshold percentiles  : {THRESHOLD_PERCENTILES}"
)

print(
    f"cache_size             : {CACHE_SIZE} MB"
)

print(
    f"shrinking              : {SHRINKING}"
)

print(
    f"train sample           : "
    f"{TRAIN_SAMPLE_LIMIT if TRAIN_SAMPLE_LIMIT is not None else 'ALL'}"
)

print(
    "Final Test             : NOT USED"
)

print()


# ============================================================
# HELPER: EVALUATE A SCORE VECTOR AT A THRESHOLD
# ============================================================

def evaluate_threshold(
    y_true,
    anomaly_score,
    threshold,
):

    prediction = np.where(
        anomaly_score >= threshold,
        1,
        0,
    )

    accuracy = accuracy_score(
        y_true,
        prediction,
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) > 0:

        fpr = fp / (fp + tn)

    else:

        fpr = 0.0

    if (tp + fn) > 0:

        attack_detection_rate = tp / (tp + fn)

    else:

        attack_detection_rate = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "attack_detection_rate": float(attack_detection_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# RUN THRESHOLD TUNING
# ============================================================

# Stage 17 intentionally keeps the Stage 16 One-Class SVM
# hyperparameters fixed and searches ONLY the anomaly-score
# detection threshold.
#
# Thresholds are calibrated from VALIDATION BENIGN scores only,
# then evaluated on the complete Validation set (BENIGN + ATTACK).
# This mirrors the threshold-selection logic used by the
# Isolation Forest pipeline.

print("=" * 80)
print("RUNNING ONE-CLASS SVM THRESHOLD TUNING")
print("=" * 80)

print(
    f"Fixed nu                : {OCSVM_BASELINE_NU}"
)

print(
    f"Fixed gamma             : {OCSVM_BASELINE_GAMMA}"
)

print(
    f"Threshold percentiles   : {THRESHOLD_PERCENTILES}"
)

print(
    "Native threshold       : 0.0"
)

print(
    "Final Test             : NOT USED"
)

print()

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "one_class_svm",
            OneClassSVM(
                kernel=KERNEL,
                nu=OCSVM_BASELINE_NU,
                gamma=OCSVM_BASELINE_GAMMA,
                cache_size=CACHE_SIZE,
                shrinking=SHRINKING,
            ),
        ),
    ]
)

print("=" * 80)
print("TRAINING STAGE 16 BASELINE OCSVM FOR THRESHOLD CALIBRATION")
print("=" * 80)

model.fit(
    X_train
)

joblib.dump(
    model,
    BEST_MODEL_FILE,
)

print(
    f"Baseline validation model saved: {BEST_MODEL_FILE}"
)

print()

anomaly_score = -model.decision_function(
    X_validation
)

validation_benign_scores = anomaly_score[
    y_validation == 0
]

if len(validation_benign_scores) == 0:

    raise RuntimeError(
        "Validation ไม่มี BENIGN samples สำหรับ Threshold Tuning"
    )

score_std = float(
    np.std(
        anomaly_score,
        ddof=0,
    )
)

score_min = float(
    np.min(anomaly_score)
)

score_max = float(
    np.max(anomaly_score)
)

print(
    f"Validation score min   : {score_min:.8f}"
)

print(
    f"Validation score max   : {score_max:.8f}"
)

print(
    f"Validation score std   : {score_std:.8f}"
)

print(
    f"Validation BENIGN score count: {len(validation_benign_scores):,}"
)

print()

# ------------------------------------------------------------
# Build threshold candidates.
# The native OCSVM boundary is included explicitly.
# Percentile thresholds are calculated from BENIGN Validation
# scores only.
# ------------------------------------------------------------

threshold_candidates = [
    {
        "threshold_percentile": 0.0,
        "threshold_type": "native",
        "threshold": 0.0,
    }
]

for threshold_percentile in THRESHOLD_PERCENTILES:

    threshold_candidates.append(
        {
            "threshold_percentile": float(
                threshold_percentile
            ),
            "threshold_type": "percentile",
            "threshold": float(
                np.percentile(
                    validation_benign_scores,
                    threshold_percentile,
                )
            ),
        }
    )

search_results = []

for threshold_info in threshold_candidates:

    threshold_percentile = float(
        threshold_info[
            "threshold_percentile"
        ]
    )

    threshold_type = str(
        threshold_info[
            "threshold_type"
        ]
    )

    threshold = float(
        threshold_info[
            "threshold"
        ]
    )

    metrics = evaluate_threshold(
        y_validation,
        anomaly_score,
        threshold,
    )

    row = {
        "nu": float(OCSVM_BASELINE_NU),
        "gamma": str(OCSVM_BASELINE_GAMMA),
        "threshold_type": threshold_type,
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "attack_detection_rate": metrics["attack_detection_rate"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "score_min": score_min,
        "score_max": score_max,
        "score_std": score_std,
    }

    search_results.append(
        row
    )

    print(
        f"Threshold {threshold_type:<10} "
        f"pct={threshold_percentile:>5.1f} "
        f"value={threshold: .8f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"FPR={metrics['fpr']:.4f}"
    )

print()

# ============================================================
# CREATE TUNING RESULT TABLE
# ============================================================

results_df = pd.DataFrame(
    search_results
)

if results_df.empty:

    raise RuntimeError(
        "ไม่พบผลลัพธ์จาก One-Class SVM Threshold Tuning"
    )

# ------------------------------------------------------------
# Apply the research protocol's Normal-only FAR constraint
# before ranking candidates.
# ------------------------------------------------------------
MAX_NORMAL_ONLY_FAR = 0.15

eligible_df = results_df[
    results_df["fpr"] <= MAX_NORMAL_ONLY_FAR
].copy()

if eligible_df.empty:
    raise RuntimeError(
        "ไม่พบ threshold ที่ผ่านเกณฑ์ Normal-only FAR <= 15%"
    )

eligible_df = eligible_df.sort_values(
    [
        "f1",
        "recall",
        "fpr",
        "precision",
    ],
    ascending=[
        False,
        False,
        True,
        False,
    ],
).reset_index(
    drop=True
)

eligible_df["validation_rank"] = (
    np.arange(1, len(eligible_df) + 1)
)

# Keep all raw candidates in the CSV, with eligibility clearly marked.
results_df["eligible_fpr_le_15pct"] = (
    results_df["fpr"] <= MAX_NORMAL_ONLY_FAR
)
results_df = results_df.sort_values(
    [
        "eligible_fpr_le_15pct",
        "f1",
        "recall",
        "fpr",
        "precision",
    ],
    ascending=[
        False,
        False,
        False,
        True,
        False,
    ],
).reset_index(drop=True)

rank_map = {
    (row["threshold_type"], float(row["threshold_percentile"])): int(rank + 1)
    for rank, (_, row) in enumerate(eligible_df.iterrows())
}
results_df["validation_rank"] = results_df.apply(
    lambda row: rank_map.get(
        (row["threshold_type"], float(row["threshold_percentile"])),
        np.nan,
    ),
    axis=1,
)

results_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# SELECT BEST CONFIGURATION
# ============================================================

# Selection is based only on Validation.
#
# Research constraint : Normal-only FAR <= 15%
# Primary criterion   : F1 maximum among eligible rows
# Tie-breaker 1       : Recall maximum
# Tie-breaker 2       : FPR minimum
# Tie-breaker 3       : Precision maximum
#
# Final Test is never used here.
# ============================================================

best_row = eligible_df.iloc[0].copy()

best_nu = float(OCSVM_BASELINE_NU)

best_gamma_raw = str(OCSVM_BASELINE_GAMMA)

best_gamma = (
    best_gamma_raw
    if best_gamma_raw == "scale"
    else float(best_gamma_raw)
)

best_threshold_percentile = float(
    best_row["threshold_percentile"]
)

best_threshold = float(
    best_row["threshold"]
)

print()
print("=" * 80)
print("BEST ONE-CLASS SVM THRESHOLD")
print("=" * 80)
print(f"Selection constraint : Normal-only FAR <= 15%")
print(f"Eligible thresholds  : {len(eligible_df):,} / {len(results_df):,}")

print(
    f"nu                    : {best_nu}"
)

print(
    f"gamma                 : {best_gamma_raw}"
)

print(
    f"threshold type        : "
    f"{best_row['threshold_type']}"
)

print(
    f"threshold percentile  : "
    f"{best_threshold_percentile:.0f}"
)

print(
    f"threshold             : "
    f"{best_threshold:.8f}"
)

print()

print(
    f"Validation Accuracy   : "
    f"{float(best_row['accuracy']):.4f}"
)

print(
    f"Validation Precision  : "
    f"{float(best_row['precision']):.4f}"
)

print(
    f"Validation Recall     : "
    f"{float(best_row['recall']):.4f}"
)

print(
    f"Validation F1         : "
    f"{float(best_row['f1']):.4f}"
)

print(
    f"Validation FPR        : "
    f"{float(best_row['fpr']):.4f}"
)

print(
    f"Attack Detection Rate: "
    f"{float(best_row['attack_detection_rate']):.4f}"
)

print()

print(
    f"TN : {int(best_row['tn']):,}"
)

print(
    f"FP : {int(best_row['fp']):,}"
)

print(
    f"FN : {int(best_row['fn']):,}"
)

print(
    f"TP : {int(best_row['tp']):,}"
)

print()


# ============================================================
# TRAIN BEST VALIDATION MODEL FOR SAVING
# ============================================================

print("=" * 80)
print("TRAINING BEST VALIDATION OCSVM")
print("=" * 80)

print(
    f"nu    : {best_nu}"
)

print(
    f"gamma : {best_gamma_raw}"
)

print()

if best_gamma_raw == "scale":

    best_model_gamma = "scale"

else:

    best_model_gamma = float(
        best_gamma_raw
    )


best_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "one_class_svm",
            OneClassSVM(
                kernel=KERNEL,
                nu=best_nu,
                gamma=best_model_gamma,
                cache_size=CACHE_SIZE,
                shrinking=SHRINKING,
            ),
        ),
    ]
)

best_model.fit(
    X_train
)

joblib.dump(
    best_model,
    BEST_MODEL_FILE,
)

print(
    f"Best validation model saved: {BEST_MODEL_FILE}"
)

print()


# ============================================================
# SAVE BEST VALIDATION SCORES
# ============================================================

best_anomaly_score = -best_model.decision_function(
    X_validation
)

best_prediction = np.where(
    best_anomaly_score >= best_threshold,
    1,
    0,
)

validation_best_result = validation_df[
    [
        "source_file",
        "window_id",
        "flow_count",
        "behavior_label",
        "attack_type",
        "target",
    ]
].copy()

validation_best_result["prediction"] = (
    best_prediction
)

validation_best_result["prediction_label"] = np.where(
    best_prediction == 1,
    "ANOMALY",
    "NORMAL",
)

validation_best_result["anomaly_score"] = (
    best_anomaly_score
)

validation_best_result["anomaly_rank"] = (
    validation_best_result["anomaly_score"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)

validation_best_result = (
    validation_best_result
    .sort_values(
        "anomaly_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

validation_best_result.to_csv(
    VALIDATION_SCORE_FILE,
    index=False,
)


# ============================================================
# SAVE BEST CONFIGURATION JSON
# ============================================================

best_config = {
    "kernel": KERNEL,
    "nu": best_nu,
    "gamma": best_gamma_raw,
    "cache_size": CACHE_SIZE,
    "shrinking": SHRINKING,
    "threshold_type": str(best_row["threshold_type"]),
    "threshold_percentile": best_threshold_percentile,
    "threshold": best_threshold,
    "validation_accuracy": float(best_row["accuracy"]),
    "validation_precision": float(best_row["precision"]),
    "validation_recall": float(best_row["recall"]),
    "validation_f1": float(best_row["f1"]),
    "validation_fpr": float(best_row["fpr"]),
    "max_allowed_normal_only_far": MAX_NORMAL_ONLY_FAR,
    "selection_constraint_satisfied": bool(float(best_row["fpr"]) <= MAX_NORMAL_ONLY_FAR),
    "validation_attack_detection_rate": float(
        best_row["attack_detection_rate"]
    ),
    "random_state": RANDOM_STATE,
    "feature_count": len(FEATURES),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "final_test_used": False,
    "selection_method": (
        "Validation F1 descending, then Recall descending, "
        "FPR ascending, Precision descending"
    ),
}

with open(
    BEST_RESULT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_config,
        f,
        indent=4,
    )

print(
    f"Best configuration saved: {BEST_RESULT_FILE}"
)

print()


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "CIC-IDS2017 ONE-CLASS SVM THRESHOLD TUNING\n"
    )

    f.write(
        "=========================================================\n"
    )

    f.write(
        "Stage 17 = One-Class SVM Hyperparameter / Threshold Tuning\n\n"
    )

    f.write(
        f"Features = {len(FEATURES)}\n"
    )

    f.write(
        f"Kernel = {KERNEL}\n"
    )

    f.write(
        f"Stage 16 fixed nu = {OCSVM_BASELINE_NU}\n"
    )

    f.write(
        f"Stage 16 fixed gamma = {OCSVM_BASELINE_GAMMA}\n"
    )

    f.write(
        "Native OCSVM threshold 0 was included in the search.\n"
    )

    f.write(
        f"threshold percentiles = {THRESHOLD_PERCENTILES}\n\n"
    )

    f.write(
        "Best Configuration\n"
    )

    f.write(
        f"nu = {best_nu}\n"
    )

    f.write(
        f"gamma = {best_gamma_raw}\n"
    )

    f.write(
        f"threshold percentile = {best_threshold_percentile:.0f}\n"
    )

    f.write(
        f"threshold = {best_threshold:.8f}\n\n"
    )

    f.write(
        "Validation Performance\n"
    )

    f.write(
        f"Accuracy = {float(best_row['accuracy']):.6f}\n"
    )

    f.write(
        f"Precision = {float(best_row['precision']):.6f}\n"
    )

    f.write(
        f"Recall = {float(best_row['recall']):.6f}\n"
    )

    f.write(
        f"F1 = {float(best_row['f1']):.6f}\n"
    )

    f.write(
        f"FPR = {float(best_row['fpr']):.6f}\n"
    )

    f.write(
        f"Attack Detection Rate = {float(best_row['attack_detection_rate']):.6f}\n\n"
    )

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        f"TN = {int(best_row['tn'])}\n"
    )

    f.write(
        f"FP = {int(best_row['fp'])}\n"
    )

    f.write(
        f"FN = {int(best_row['fn'])}\n"
    )

    f.write(
        f"TP = {int(best_row['tp'])}\n\n"
    )

    f.write(
        "Methodology\n"
    )

    f.write(
        "Feature selection comes from Stage 08 and is TRAIN-only.\n"
    )

    f.write(
        "This stage trains only on BENIGN Train behavior.\n"
    )

    f.write(
        "Only threshold selection is performed in Stage 17.\n"
    )

    f.write(
        "Final Test is NOT used in Stage 17.\n"
    )

    f.write(
        "The selected threshold is saved for Stage 18 and later Final OCSVM stages.\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 80)
print("ONE-CLASS SVM THRESHOLD TUNING COMPLETE")
print("=" * 80)

print(
    f"Threshold result file : {RESULT_FILE}"
)

print(
    f"Baseline model file : {BEST_MODEL_FILE}"
)

print(
    f"Threshold config file: {BEST_RESULT_FILE}"
)

print(
    f"Validation score file: {VALIDATION_SCORE_FILE}"
)

print(
    f"Summary file       : {SUMMARY_FILE}"
)

print()

print(
    f"Fixed nu            : {best_nu}"
)

print(
    f"Fixed gamma         : {best_gamma_raw}"
)

print(
    f"Best threshold pct : {best_threshold_percentile:.0f}"
)

print(
    f"Best threshold     : {best_threshold:.8f}"
)

print(
    f"Best Validation F1 : {float(best_row['f1']):.4f}\n"
    f"Normal-only FAR     : {float(best_row['fpr']):.4f}\n"
    f"FAR constraint <=15%: {'PASS' if float(best_row['fpr']) <= MAX_NORMAL_ONLY_FAR else 'FAIL'}"
)

print()

print(
    "Stage 17 used Validation only. "
    "Final Test was not used."
)

print(
    "Ready for Stage 18 One-Class SVM hyperparameter tuning."
)

print("=" * 80)
