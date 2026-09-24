from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# CIC-IDS2017
# STAGE 40 — STATISTICAL THRESHOLD BASELINE
# ============================================================
#
# Purpose:
#   Add an independent Statistical Threshold baseline to the
#   CIC-IDS2017 pipeline so it can be compared with:
#       - Isolation Forest
#       - One-Class SVM
#       - Local Outlier Factor
#
# Protocol:
#   1. Train BENIGN only -> build statistical / robust deviation
#      reference.
#   2. Validation -> select percentile/threshold.
#   3. Lock selected configuration.
#   4. Final Test -> evaluation only.
#
# IMPORTANT:
#   This script does NOT reuse Isolation Forest / OCSVM / LOF
#   anomaly scores.
#
#   The available thesis source specifies that Statistical
#   Threshold uses statistics + robust deviation of each feature,
#   and that percentile is selected using Validation. The exact
#   original implementation formula was not available in the
#   retrieved source files, so this CIC implementation uses an
#   explicit robust MAD-based deviation score.
#
# Input:
#   train_behavior.csv
#   validation_behavior.csv
#   test_behavior.csv
#   selected_behavior_features.csv
#
# Output:
#   statistical_threshold_validation.csv
#   statistical_threshold_final_predictions.csv
#   statistical_threshold_final_metrics.json
#   statistical_threshold_final_summary.txt
#   statistical_threshold_config.json
#   statistical_threshold_scores.csv
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_behavior.csv"
VALIDATION_FILE = BASE_DIR / "validation_behavior.csv"
TEST_FILE = BASE_DIR / "test_behavior.csv"
FEATURE_FILE = BASE_DIR / "selected_behavior_features.csv"

VALIDATION_RESULT_FILE = (
    BASE_DIR / "statistical_threshold_validation.csv"
)

PREDICTION_FILE = (
    BASE_DIR / "statistical_threshold_final_predictions.csv"
)

METRICS_FILE = (
    BASE_DIR / "statistical_threshold_final_metrics.json"
)

SUMMARY_FILE = (
    BASE_DIR / "statistical_threshold_final_summary.txt"
)

CONFIG_FILE = (
    BASE_DIR / "statistical_threshold_config.json"
)

SCORES_FILE = (
    BASE_DIR / "statistical_threshold_scores.csv"
)

# Percentiles are selected using Validation.
# 99 is included because the custom dataset used percentile=99.
PERCENTILES = [90, 92, 95, 97, 99]

RANDOM_STATE = 42
EPS = 1e-9


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {path}"
        )


def load_features():
    features_df = pd.read_csv(FEATURE_FILE)

    if "feature" not in features_df.columns:
        raise ValueError(
            "selected_behavior_features.csv "
            "ต้องมี column ชื่อ 'feature'"
        )

    features = (
        features_df["feature"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))

    # flow_count is metadata in the current CIC pipeline.
    if "flow_count" in features:
        features.remove("flow_count")

    return features


def prepare_numeric(df, features):
    x = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values using the training reference later.
    return x


def fit_robust_reference(train_x):
    """
    Build robust statistics from BENIGN TRAIN only.

    Median:
        center of the feature distribution.

    MAD:
        median absolute deviation.

    Robust scale:
        1.4826 * MAD, making it comparable to standard deviation
        under a normal distribution.

    If MAD is zero, fall back to IQR-based scale and finally 1.
    """

    medians = train_x.median()

    abs_dev = (train_x - medians).abs()
    mad = abs_dev.median()

    robust_scale = 1.4826 * mad

    q1 = train_x.quantile(0.25)
    q3 = train_x.quantile(0.75)

    iqr_scale = (q3 - q1) / 1.349

    robust_scale = robust_scale.where(
        robust_scale > EPS,
        iqr_scale
    )

    robust_scale = robust_scale.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(1.0)

    robust_scale = robust_scale.clip(
        lower=EPS
    )

    return medians, robust_scale


def robust_anomaly_score(
    x,
    medians,
    robust_scale
):
    """
    Calculate a statistical anomaly score.

    For every feature:

        robust_z =
            abs(x - median) / robust_scale

    The final score is the mean robust deviation across the
    selected behavior features.

    Higher score = more statistically unusual.
    """

    x = x.copy()

    x = x.fillna(medians)

    robust_z = (
        x.subtract(medians)
        .abs()
        .divide(robust_scale)
    )

    robust_z = robust_z.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    score = robust_z.mean(axis=1)

    return score.to_numpy()


def make_threshold(train_scores, percentile):
    return float(
        np.percentile(
            train_scores,
            percentile
        )
    )


def evaluate_scores(
    scores,
    y_true,
    threshold,
    frame=None
):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    benign_mask = (
        y_true == 0
    )

    if benign_mask.sum() > 0:
        overall_far = float(
            fp / benign_mask.sum()
        )
    else:
        overall_far = 0.0

    normal_only_far = np.nan

    # CIC has no Flash Crowd class. If traffic_type exists and
    # Normal is available, calculate Normal-only FAR.
    if frame is not None and "traffic_type" in frame.columns:
        traffic = (
            frame["traffic_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        normal_mask = (
            benign_mask
            & traffic.eq("normal").to_numpy()
        )

        if normal_mask.sum() > 0:
            normal_only_far = float(
                y_pred[normal_mask].sum()
                / normal_mask.sum()
            )

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "overall_far": float(overall_far),
        "normal_only_far": (
            float(normal_only_far)
            if not np.isnan(normal_only_far)
            else np.nan
        ),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def select_best_validation(result_df):
    """
    Select configuration from Validation only.

    Primary:
        highest F1

    Tie-break:
        higher Recall
        lower FAR
        higher Precision
    """

    ranked = result_df.sort_values(
        by=[
            "f1",
            "recall",
            "overall_far",
            "precision",
        ],
        ascending=[
            False,
            False,
            True,
            False,
        ]
    )

    return ranked.iloc[0].to_dict()


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    FEATURE_FILE,
]:
    require_file(path)


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("STAGE 40 — CIC-IDS2017 STATISTICAL THRESHOLD")
print("=" * 100)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

features = load_features()

print()
print(f"Train rows      : {len(train_df):,}")
print(f"Validation rows : {len(validation_df):,}")
print(f"Final Test rows : {len(test_df):,}")
print(f"Features used   : {len(features)}")
print()

print("FEATURES")
for feature in features:
    print(f"  - {feature}")

print()


# ============================================================
# TARGET
# ============================================================

for name, df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Final Test", test_df),
]:
    if "target" not in df.columns:
        raise ValueError(
            f"{name} ไม่มี column 'target'"
        )

train_target = (
    pd.to_numeric(
        train_df["target"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

validation_target = (
    pd.to_numeric(
        validation_df["target"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

test_target = (
    pd.to_numeric(
        test_df["target"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# TRAIN MUST BE BENIGN ONLY
# ============================================================

if (train_target == 1).sum() != 0:
    raise ValueError(
        "Train ต้องมีเฉพาะ BENIGN แต่พบ ATTACK"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

train_x = prepare_numeric(
    train_df,
    features
)

validation_x = prepare_numeric(
    validation_df,
    features
)

test_x = prepare_numeric(
    test_df,
    features
)


# ============================================================
# FIT STATISTICAL REFERENCE FROM TRAIN BENIGN ONLY
# ============================================================

print("=" * 100)
print("BUILD ROBUST STATISTICAL REFERENCE")
print("=" * 100)

medians, robust_scale = fit_robust_reference(
    train_x
)

# Fill missing values using training medians.
train_x = train_x.fillna(medians)
validation_x = validation_x.fillna(medians)
test_x = test_x.fillna(medians)


# ============================================================
# CREATE INDEPENDENT STATISTICAL SCORES
# ============================================================

train_scores = robust_anomaly_score(
    train_x,
    medians,
    robust_scale
)

validation_scores = robust_anomaly_score(
    validation_x,
    medians,
    robust_scale
)

test_scores = robust_anomaly_score(
    test_x,
    medians,
    robust_scale
)

print(
    f"Train score range      : "
    f"{train_scores.min():.8f} - "
    f"{train_scores.max():.8f}"
)

print(
    f"Validation score range : "
    f"{validation_scores.min():.8f} - "
    f"{validation_scores.max():.8f}"
)

print(
    f"Final Test score range : "
    f"{test_scores.min():.8f} - "
    f"{test_scores.max():.8f}"
)

print()


# ============================================================
# VALIDATION THRESHOLD TUNING
# ============================================================

print("=" * 100)
print("VALIDATION — STATISTICAL THRESHOLD TUNING")
print("=" * 100)

validation_rows = []

for percentile in PERCENTILES:

    threshold = make_threshold(
        train_scores,
        percentile
    )

    metrics = evaluate_scores(
        validation_scores,
        validation_target,
        threshold,
        validation_df
    )

    row = {
        "percentile": percentile,
        **metrics,
    }

    validation_rows.append(row)

    print(
        f"P{percentile:<3} | "
        f"threshold={threshold:.8f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"FAR={metrics['overall_far']:.4f}"
    )

validation_result = pd.DataFrame(
    validation_rows
)

validation_result.to_csv(
    VALIDATION_RESULT_FILE,
    index=False,
    float_format="%.10f"
)


# ============================================================
# SELECT CONFIGURATION FROM VALIDATION ONLY
# ============================================================

best = select_best_validation(
    validation_result
)

best_percentile = int(
    best["percentile"]
)

locked_threshold = float(
    best["threshold"]
)

print()
print("=" * 100)
print("SELECTED CONFIGURATION")
print("=" * 100)

print(
    f"Selected percentile : P{best_percentile}"
)

print(
    f"Locked threshold    : "
    f"{locked_threshold:.10f}"
)

print(
    f"Validation F1       : "
    f"{best['f1']:.4f}"
)

print()


# ============================================================
# FINAL TEST — LOCKED CONFIGURATION
# ============================================================

print("=" * 100)
print("FINAL TEST — STATISTICAL THRESHOLD")
print("=" * 100)

final_metrics = evaluate_scores(
    test_scores,
    test_target,
    locked_threshold,
    test_df
)

print(
    f"Accuracy   : {final_metrics['accuracy']:.4f}"
)

print(
    f"Precision  : {final_metrics['precision']:.4f}"
)

print(
    f"Recall     : {final_metrics['recall']:.4f}"
)

print(
    f"F1         : {final_metrics['f1']:.4f}"
)

print(
    f"Overall FAR: {final_metrics['overall_far']:.4f}"
)

print(
    f"ROC-AUC    : {final_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC     : {final_metrics['pr_auc']:.4f}"
)

print()
print("CONFUSION MATRIX")
print(
    f"TN={final_metrics['tn']:,}, "
    f"FP={final_metrics['fp']:,}, "
    f"FN={final_metrics['fn']:,}, "
    f"TP={final_metrics['tp']:,}"
)

print()


# ============================================================
# SAVE FINAL PREDICTIONS
# ============================================================

predictions = test_df.copy()

predictions["statistical_anomaly_score"] = test_scores

predictions["statistical_threshold"] = (
    locked_threshold
)

predictions["anomaly_flag"] = (
    test_scores >= locked_threshold
).astype(int)

predictions.to_csv(
    PREDICTION_FILE,
    index=False,
    float_format="%.10f"
)


# ============================================================
# SAVE ALL SCORES
# ============================================================

train_score_df = train_df.copy()
train_score_df["split"] = "TRAIN"
train_score_df["statistical_anomaly_score"] = train_scores
train_score_df["anomaly_flag"] = (
    train_scores >= locked_threshold
).astype(int)

validation_score_df = validation_df.copy()
validation_score_df["split"] = "VALIDATION"
validation_score_df["statistical_anomaly_score"] = (
    validation_scores
)
validation_score_df["anomaly_flag"] = (
    validation_scores >= locked_threshold
).astype(int)

test_score_df = test_df.copy()
test_score_df["split"] = "FINAL_TEST"
test_score_df["statistical_anomaly_score"] = test_scores
test_score_df["anomaly_flag"] = (
    test_scores >= locked_threshold
).astype(int)

score_output = pd.concat(
    [
        train_score_df,
        validation_score_df,
        test_score_df,
    ],
    ignore_index=True
)

score_output.to_csv(
    SCORES_FILE,
    index=False,
    float_format="%.10f"
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "model": "Statistical Threshold",
    "score_method": (
        "mean absolute robust deviation "
        "using median and MAD"
    ),
    "robust_scale": "1.4826 * MAD with IQR fallback",
    "percentiles_tested": PERCENTILES,
    "selected_percentile": best_percentile,
    "locked_threshold": locked_threshold,
    "selection_data": "Validation only",
    "final_test_used_for_tuning": False,
    "final_test_used_for_retraining": False,
    "features": features,
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_json = {
    "model": "Statistical Threshold",
    "test_rows": int(len(test_df)),
    "percentile": best_percentile,
    "threshold": locked_threshold,
    "accuracy": final_metrics["accuracy"],
    "precision": final_metrics["precision"],
    "recall": final_metrics["recall"],
    "f1": final_metrics["f1"],
    "overall_far": final_metrics["overall_far"],
    "normal_only_far": final_metrics["normal_only_far"],
    "roc_auc": final_metrics["roc_auc"],
    "pr_auc": final_metrics["pr_auc"],
    "tn": final_metrics["tn"],
    "fp": final_metrics["fp"],
    "fn": final_metrics["fn"],
    "tp": final_metrics["tp"],
    "final_test_used_for_tuning": False,
    "final_test_used_for_retraining": False,
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics_json,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary = []

summary.append("=" * 100)
summary.append("STAGE 40 — CIC-IDS2017 STATISTICAL THRESHOLD")
summary.append("=" * 100)
summary.append("")

summary.append(
    "Statistical score = mean absolute robust deviation "
    "from TRAIN BENIGN statistics."
)

summary.append(
    "Reference statistics were fitted using TRAIN BENIGN only."
)

summary.append(
    "Percentile/threshold was selected using VALIDATION only."
)

summary.append(
    "FINAL TEST was used only for final evaluation."
)

summary.append("")

summary.append(
    f"Selected percentile : P{best_percentile}"
)

summary.append(
    f"Locked threshold    : {locked_threshold:.10f}"
)

summary.append("")

summary.append("FINAL TEST METRICS")
summary.append(
    f"Accuracy    : {final_metrics['accuracy']:.6f}"
)
summary.append(
    f"Precision   : {final_metrics['precision']:.6f}"
)
summary.append(
    f"Recall      : {final_metrics['recall']:.6f}"
)
summary.append(
    f"F1          : {final_metrics['f1']:.6f}"
)
summary.append(
    f"Overall FAR : {final_metrics['overall_far']:.6f}"
)
summary.append(
    f"ROC-AUC     : {final_metrics['roc_auc']:.6f}"
)
summary.append(
    f"PR-AUC      : {final_metrics['pr_auc']:.6f}"
)

summary.append("")
summary.append("CONFUSION MATRIX")
summary.append(
    f"TN={final_metrics['tn']:,}, "
    f"FP={final_metrics['fp']:,}, "
    f"FN={final_metrics['fn']:,}, "
    f"TP={final_metrics['tp']:,}"
)

summary.append("")
summary.append(
    "Final Test used for tuning: NO"
)
summary.append(
    "Final Test used for retraining: NO"
)

SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8"
)


# ============================================================
# DONE
# ============================================================

print("=" * 100)
print("STAGE 40 COMPLETED")
print("=" * 100)

print()
print("Created files:")
print(f"  {VALIDATION_RESULT_FILE.name}")
print(f"  {PREDICTION_FILE.name}")
print(f"  {METRICS_FILE.name}")
print(f"  {SUMMARY_FILE.name}")
print(f"  {CONFIG_FILE.name}")
print(f"  {SCORES_FILE.name}")
print()
print(
    "Final Test was used ONLY for final evaluation."
)
