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
    classification_report,
)

# ============================================================
# STAGE 15: FINAL ISOLATION FOREST EVALUATION ON FINAL TEST
# ============================================================
# Final Test is used ONLY here for final evaluation/reporting.
# Threshold was selected from Validation in Stage 10/11.
# No test-derived rank threshold is used.
# ============================================================

BASE = Path(__file__).resolve().parent
TEST = BASE / "test_behavior.csv"
SCORE = BASE / "anomaly_scores.csv"
CFG = BASE / "behavior_tuning_config.json"
OUT = BASE / "final_behavior_metrics.csv"
SUMMARY = BASE / "final_behavior_analysis_summary.txt"
ATT = BASE / "attack_type_analysis.csv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    for path in [TEST, SCORE, CFG]:
        require_file(path)

    test = pd.read_csv(TEST)
    score = pd.read_csv(SCORE)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))

    if len(test) != len(score):
        raise ValueError(
            f"Final Test rows ({len(test)}) และ anomaly score rows ({len(score)}) ไม่ตรงกัน"
        )

    if "target" not in test.columns:
        raise ValueError("Final Test ต้องมี column 'target'")
    if "anomaly_score_raw" not in score.columns:
        raise ValueError("anomaly_scores.csv ต้องมี column 'anomaly_score_raw'")

    threshold = float(cfg["threshold"])
    threshold_percentile = float(cfg.get("threshold_percentile", np.nan))

    y_true = test["target"].astype(int).to_numpy()
    scores = score["anomaly_score_raw"].astype(float).to_numpy()
    prediction = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, prediction)
    precision = precision_score(y_true, prediction, zero_division=0)
    recall = recall_score(y_true, prediction, zero_division=0)
    f1 = f1_score(y_true, prediction, zero_division=0)
    overall_far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    normal_only_far = overall_far
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan

    metrics = {
        "model": "Isolation Forest",
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overall_far": overall_far,
        "normal_only_far": normal_only_far,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "test_rows": int(len(test)),
        "test_benign": int((y_true == 0).sum()),
        "test_attack": int((y_true == 1).sum()),
        "predicted_normal": int((prediction == 0).sum()),
        "predicted_anomaly": int((prediction == 1).sum()),
    }

    pd.DataFrame([metrics]).to_csv(OUT, index=False)

    # --------------------------------------------------------
    # Attack-type analysis
    # --------------------------------------------------------
    attack_table = pd.DataFrame()
    if "attack_type" in test.columns:
        attack_table = test.copy()
        attack_table["prediction"] = prediction
        attack_table["score"] = scores
        attack_table = attack_table[attack_table["target"] == 1].copy()

        if not attack_table.empty:
            attack_table = (
                attack_table
                .groupby("attack_type", dropna=False)
                .agg(
                    samples=("target", "size"),
                    detected=("prediction", "sum"),
                    mean_score=("score", "mean"),
                )
                .reset_index()
            )
            attack_table["missed"] = attack_table["samples"] - attack_table["detected"]
            attack_table["detection_rate"] = attack_table["detected"] / attack_table["samples"]
            attack_table = attack_table.sort_values("detection_rate", ascending=False)
            attack_table.to_csv(ATT, index=False)

    # --------------------------------------------------------
    # Human-readable summary file
    # --------------------------------------------------------
    report_lines = [
        "CIC-IDS2017 — ISOLATION FOREST FINAL TEST ANALYSIS",
        "=" * 78,
        "",
        "[1] FINAL TEST",
        f"Test rows        : {len(test):,}",
        f"BENIGN           : {(y_true == 0).sum():,}",
        f"ATTACK           : {(y_true == 1).sum():,}",
        "",
        "[2] LOCKED CONFIGURATION",
        f"n_estimators     : {cfg.get('n_estimators', 'from Stage 10/11')}",
        f"max_samples      : {cfg.get('max_samples', 'from Stage 11')}",
        f"max_features     : {cfg.get('max_features', 'from Stage 11')}",
        f"contamination    : {cfg.get('contamination', 0.01)}",
        f"threshold source : Validation",
        f"threshold pct    : {threshold_percentile:.0f}" if not np.isnan(threshold_percentile) else "threshold pct    : N/A",
        f"threshold        : {threshold:.10f}",
        "",
        "[3] FINAL TEST PERFORMANCE",
        f"Accuracy         : {accuracy:.4f}",
        f"Precision        : {precision:.4f}",
        f"Recall           : {recall:.4f}",
        f"F1 Score         : {f1:.4f}",
        f"Overall FAR      : {pct(overall_far)}",
        f"Normal-only FAR  : {pct(normal_only_far)}",
        f"ROC-AUC          : {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC          : N/A",
        f"PR-AUC            : {pr_auc:.4f}" if not np.isnan(pr_auc) else "PR-AUC            : N/A",
        "",
        "[4] PREDICTION SUMMARY",
        f"Predicted Normal : {(prediction == 0).sum():,}",
        f"Predicted Anomaly: {(prediction == 1).sum():,}",
        "",
        "[5] CONFUSION MATRIX",
        "                  Pred Normal   Pred Attack",
        f"Actual BENIGN      {tn:>10,}    {fp:>10,}",
        f"Actual ATTACK      {fn:>10,}    {tp:>10,}",
        "",
        "[6] CLASSIFICATION REPORT",
        classification_report(
            y_true,
            prediction,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            digits=4,
            zero_division=0,
        ),
        "",
        "[7] DATA USAGE CHECK",
        "Final Test used for evaluation : YES",
        "Final Test used for tuning     : NO",
        "Threshold source               : Validation",
        "Score calibration source       : Train BENIGN only",
    ]

    if not attack_table.empty:
        report_lines.extend([
            "",
            "[8] ATTACK-TYPE ANALYSIS",
            attack_table.to_string(index=False),
        ])

    SUMMARY.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print("=" * 90)
    print("CIC-IDS2017 — STAGE 15: FINAL ISOLATION FOREST TEST EVALUATION")
    print("=" * 90)
    print()

    print("[FINAL TEST]")
    print(f"  Total rows       : {len(test):,}")
    print(f"  BENIGN           : {(y_true == 0).sum():,}")
    print(f"  ATTACK           : {(y_true == 1).sum():,}")
    print()

    print("[LOCKED CONFIGURATION — FROM VALIDATION]")
    print(f"  n_estimators     : {cfg.get('n_estimators', 'N/A')}")
    print(f"  max_samples      : {cfg.get('max_samples', 'N/A')}")
    print(f"  max_features     : {cfg.get('max_features', 'N/A')}")
    print(f"  contamination    : {cfg.get('contamination', 'N/A')}")
    print(f"  threshold pct    : {threshold_percentile:.0f}" if not np.isnan(threshold_percentile) else "  threshold pct    : N/A")
    print(f"  threshold        : {threshold:.10f}")
    print()

    print("[FINAL TEST PERFORMANCE]")
    performance = pd.DataFrame([
        ["Accuracy", accuracy],
        ["Precision", precision],
        ["Recall", recall],
        ["F1 Score", f1],
        ["Overall FAR", overall_far],
        ["Normal-only FAR", normal_only_far],
        ["ROC-AUC", roc_auc],
        ["PR-AUC", pr_auc],
    ], columns=["Metric", "Value"])
    performance["Value"] = performance["Value"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    print(performance.to_string(index=False))
    print()

    print("[CONFUSION MATRIX]")
    print("                  Pred Normal   Pred Attack")
    print(f"Actual BENIGN      {tn:>10,}    {fp:>10,}")
    print(f"Actual ATTACK      {fn:>10,}    {tp:>10,}")
    print()

    print("[PREDICTION SUMMARY]")
    print(f"  Predicted Normal : {(prediction == 0).sum():,}")
    print(f"  Predicted Anomaly: {(prediction == 1).sum():,}")
    print()

    if not attack_table.empty:
        print("[ATTACK-TYPE DETECTION]")
        display_attack = attack_table.copy()
        display_attack["detection_rate"] = display_attack["detection_rate"].map(lambda x: f"{x * 100:.2f}%")
        display_attack["mean_score"] = display_attack["mean_score"].map(lambda x: f"{x:.6f}")
        print(display_attack.to_string(index=False))
        print()

    print("[DATA USAGE]")
    print("  Final Test used for evaluation : YES")
    print("  Final Test used for tuning    : NO")
    print("  Threshold source              : Validation")
    print("  Score calibration source      : Train BENIGN only")
    print()

    print("[SAVED FILES]")
    print(f"  {OUT}")
    print(f"  {SUMMARY}")
    if not attack_table.empty:
        print(f"  {ATT}")
    print("=" * 90)
    print("STAGE 15 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
