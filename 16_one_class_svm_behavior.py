# ============================================================

from pathlib import Path

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
# CIC-IDS2017 -> ONE-CLASS SVM BEHAVIOR MODEL
# ============================================================
#
# Purpose:
#   Build a second anomaly-detection model using One-Class SVM
#   on the SAME Behavior Dataset and the SAME selected features
#   used by the Isolation Forest pipeline.
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
#   Stage 16 is a BASELINE One-Class SVM model.
#   Hyperparameter tuning will be performed in a later stage
#   using Validation only.
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
# Model:
#   Kernel = RBF
#   nu     = 0.01
#   gamma  = scale
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

MODEL_FILE = (
    BASE_DIR / "one_class_svm_behavior_baseline.pkl"
)

VALIDATION_SCORE_FILE = (
    BASE_DIR / "one_class_svm_validation_scores.csv"
)

SUMMARY_FILE = (
    BASE_DIR / "one_class_svm_baseline_summary.txt"
)

RANDOM_STATE = 42

# ------------------------------------------------------------
# Baseline One-Class SVM parameters
# ------------------------------------------------------------

KERNEL = "rbf"

NU = 0.01

GAMMA = "scale"

CACHE_SIZE = 4096

SHRINKING = True

# Set to None to use ALL BENIGN Train rows.
#
# One-Class SVM with an RBF kernel can be computationally heavy
# on large datasets. If training becomes too slow or memory-heavy,
# this can be changed to a fixed deterministic sample size, e.g.
# 10000, for an explicitly documented computational experiment.
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
# BASELINE PARAMETERS
# ============================================================

print("=" * 80)
print("BASELINE ONE-CLASS SVM PARAMETERS")
print("=" * 80)

print(
    f"kernel       : {KERNEL}"
)

print(
    f"nu           : {NU}"
)

print(
    f"gamma        : {GAMMA}"
)

print(
    f"cache_size   : {CACHE_SIZE} MB"
)

print(
    f"shrinking    : {SHRINKING}"
)

print(
    f"train sample : "
    f"{TRAIN_SAMPLE_LIMIT if TRAIN_SAMPLE_LIMIT is not None else 'ALL'}"
)

print(
    "Final Test   : NOT USED"
)

print()


# ============================================================
# BUILD ONE-CLASS SVM MODEL
# ============================================================

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
                nu=NU,
                gamma=GAMMA,
                cache_size=CACHE_SIZE,
                shrinking=SHRINKING,
            ),
        ),
    ]
)


# ============================================================
# TRAIN BASELINE MODEL
# ============================================================

print("=" * 80)
print("TRAINING ONE-CLASS SVM")
print("=" * 80)

model.fit(
    X_train
)

print(
    "One-Class SVM training complete."
)

print()


# ============================================================
# SAVE BASELINE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE,
)

print(
    f"Model saved: {MODEL_FILE}"
)

print()


# ============================================================
# VALIDATION PREDICTION
# ============================================================

print("=" * 80)
print("VALIDATION EVALUATION")
print("=" * 80)

raw_prediction = model.predict(
    X_validation
)

# OneClassSVM:
#    1  = inlier / normal
#   -1  = outlier / anomaly
prediction = np.where(
    raw_prediction == -1,
    1,
    0,
)

# decision_function:
#   higher = more normal
#   lower  = more anomalous
#
# Convert to the project's common convention:
#   higher anomaly_score = more anomalous
anomaly_score = -model.decision_function(
    X_validation
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_validation,
    prediction,
)

precision = precision_score(
    y_validation,
    prediction,
    zero_division=0,
)

recall = recall_score(
    y_validation,
    prediction,
    zero_division=0,
)

f1 = f1_score(
    y_validation,
    prediction,
    zero_division=0,
)

cm = confusion_matrix(
    y_validation,
    prediction,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

if (tn + fp) > 0:

    fpr = fp / (fp + tn)

else:

    fpr = 0.0

if (tp + fn) > 0:

    attack_detection_rate = (
        tp / (tp + fn)
    )

else:

    attack_detection_rate = 0.0


# ============================================================
# PRINT METRICS
# ============================================================

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"False Positive Rate : {fpr:.4f}"
)

print(
    f"Attack Detection Rate : "
    f"{attack_detection_rate:.4f}"
)

print()

print("Confusion Matrix")

print(cm)

print()

print(
    f"TN : {tn:,}"
)

print(
    f"FP : {fp:,}"
)

print(
    f"FN : {fn:,}"
)

print(
    f"TP : {tp:,}"
)

print()


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

predicted_normal = (
    prediction == 0
).sum()

predicted_anomaly = (
    prediction == 1
).sum()

actual_benign = (
    y_validation == 0
).sum()

actual_attack = (
    y_validation == 1
).sum()

print("=" * 80)
print("PREDICTION DISTRIBUTION")
print("=" * 80)

print(
    f"Predicted Normal  : "
    f"{predicted_normal:,}"
)

print(
    f"Predicted Anomaly : "
    f"{predicted_anomaly:,}"
)

print()

print(
    f"Actual BENIGN     : "
    f"{actual_benign:,}"
)

print(
    f"Actual ATTACK     : "
    f"{actual_attack:,}"
)

print()


# ============================================================
# ANOMALY SCORE SUMMARY
# ============================================================

print("=" * 80)
print("VALIDATION ANOMALY SCORE SUMMARY")
print("=" * 80)

print(
    f"Min    : {anomaly_score.min():.6f}"
)

print(
    f"Max    : {anomaly_score.max():.6f}"
)

print(
    f"Mean   : {anomaly_score.mean():.6f}"
)

print(
    f"Median : {np.median(anomaly_score):.6f}"
)

print(
    f"Std    : {anomaly_score.std(ddof=0):.6f}"
)

print()


# ============================================================
# SCORE BY TARGET
# ============================================================

validation_result = validation_df[
    [
        "source_file",
        "window_id",
        "flow_count",
        "behavior_label",
        "attack_type",
        "target",
    ]
].copy()

validation_result["prediction"] = (
    prediction
)

validation_result["prediction_label"] = np.where(
    prediction == 1,
    "ANOMALY",
    "NORMAL",
)

validation_result["anomaly_score"] = (
    anomaly_score
)


# ============================================================
# VALIDATION SCORE ANALYSIS
# ============================================================

print("=" * 80)
print("SCORE BY TARGET")
print("=" * 80)

for target_value, label in [
    (0, "BENIGN"),
    (1, "ATTACK"),
]:

    values = validation_result.loc[
        validation_result["target"] == target_value,
        "anomaly_score",
    ]

    if len(values) == 0:

        continue

    print()
    print(label)
    print(
        f"Count  : {len(values):,}"
    )
    print(
        f"Mean   : {values.mean():.6f}"
    )
    print(
        f"Median : {values.median():.6f}"
    )
    print(
        f"Min    : {values.min():.6f}"
    )
    print(
        f"Max    : {values.max():.6f}"
    )

print()


# ============================================================
# SAVE VALIDATION SCORES
# ============================================================

validation_result = (
    validation_result
    .sort_values(
        "anomaly_score",
        ascending=False,
    )
    .reset_index(drop=True)
)

validation_result["anomaly_rank"] = (
    validation_result["anomaly_score"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)

validation_result.to_csv(
    VALIDATION_SCORE_FILE,
    index=False,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "CIC-IDS2017 ONE-CLASS SVM BEHAVIOR MODEL\n"
    )

    f.write(
        "=============================================\n"
    )

    f.write(
        "Stage 16 = BASELINE MODEL\n\n"
    )

    f.write(
        f"Features = {len(FEATURES)}\n"
    )

    f.write(
        f"Kernel = {KERNEL}\n"
    )

    f.write(
        f"nu = {NU}\n"
    )

    f.write(
        f"gamma = {GAMMA}\n"
    )

    f.write(
        f"cache_size = {CACHE_SIZE}\n"
    )

    f.write(
        f"shrinking = {SHRINKING}\n"
    )

    f.write(
        f"train_sample_limit = {TRAIN_SAMPLE_LIMIT}\n\n"
    )

    f.write(
        "Training\n"
    )

    f.write(
        f"Train BENIGN = "
        f"{(train_target == 0).sum()}\n"
    )

    f.write(
        f"Train ATTACK = "
        f"{(train_target == 1).sum()}\n\n"
    )

    f.write(
        "Validation\n"
    )

    f.write(
        f"Validation BENIGN = "
        f"{validation_benign_count}\n"
    )

    f.write(
        f"Validation ATTACK = "
        f"{validation_attack_count}\n"
    )

    f.write(
        f"Accuracy = {accuracy:.6f}\n"
    )

    f.write(
        f"Precision = {precision:.6f}\n"
    )

    f.write(
        f"Recall = {recall:.6f}\n"
    )

    f.write(
        f"F1 = {f1:.6f}\n"
    )

    f.write(
        f"FPR = {fpr:.6f}\n"
    )

    f.write(
        f"Attack Detection Rate = "
        f"{attack_detection_rate:.6f}\n\n"
    )

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        f"TN = {tn}\n"
    )

    f.write(
        f"FP = {fp}\n"
    )

    f.write(
        f"FN = {fn}\n"
    )

    f.write(
        f"TP = {tp}\n\n"
    )

    f.write(
        "Anomaly Score Convention\n"
    )

    f.write(
        "anomaly_score = -decision_function()\n"
    )

    f.write(
        "Higher score = more anomalous\n\n"
    )

    f.write(
        "Important Methodology\n"
    )

    f.write(
        "Feature selection comes from Stage 08 and is TRAIN-only.\n"
    )

    f.write(
        "This stage trains only on BENIGN Train behavior.\n"
    )

    f.write(
        "This stage evaluates on Validation only.\n"
    )

    f.write(
        "Final Test is NOT used in Stage 16.\n"
    )

    f.write(
        "Hyperparameter tuning will be performed in a later stage.\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 80)
print("ONE-CLASS SVM BASELINE COMPLETE")
print("=" * 80)

print(
    f"Model file      : {MODEL_FILE}"
)

print(
    f"Validation file : {VALIDATION_SCORE_FILE}"
)

print(
    f"Summary file    : {SUMMARY_FILE}"
)

print()

print(
    "Stage 16 used Validation only. "
    "Final Test was not used."
)

print(
    "Ready for One-Class SVM hyperparameter / threshold tuning."
)

print("=" * 80)
