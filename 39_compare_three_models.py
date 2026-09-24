from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# ============================================================
# STAGE 39 — FINAL COMPARISON OF 3 MODELS
# Models:
#   1. Isolation Forest
#   2. One-Class SVM
#   3. Local Outlier Factor
#
# This stage ONLY compares already-finished Final Test results.
# No retraining / no tuning.
# ============================================================

BASE = Path(__file__).resolve().parent

# Final metric files
IF_METRICS = BASE / "final_behavior_metrics.csv"
OCSVM_METRICS = BASE / "final_one_class_svm_final_metrics.csv"
LOF_METRICS = BASE / "final_lof_final_metrics.csv"

# IF fallback / score sources
IF_COMBINED = BASE / "model_comparison.csv"
IF_SCORE = BASE / "anomaly_scores.csv"

OUTPUT_CSV = BASE / "final_three_model_comparison.csv"
OUTPUT_TXT = BASE / "final_three_model_comparison_summary.txt"
OUTPUT_PNG = BASE / "final_three_model_metrics.png"

CORE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")


def norm(df):
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def numeric_value(row, names):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return float(row[n])
    return None


def derive_far(row):
    far = numeric_value(row, ["overall_far", "fpr", "far"])
    if far is not None:
        return far

    tn = numeric_value(row, ["tn"])
    fp = numeric_value(row, ["fp"])
    if tn is not None and fp is not None and (tn + fp) > 0:
        return fp / (tn + fp)

    return np.nan


def derive_pr_auc(row, score_file=None):
    pr = numeric_value(row, ["pr_auc", "average_precision"])
    if pr is not None:
        return pr

    if score_file is not None and score_file.exists():
        df = norm(pd.read_csv(score_file))

        # Try target + anomaly score
        if "target" in df.columns and "anomaly_score" in df.columns:
            y = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int).to_numpy()
            s = pd.to_numeric(df["anomaly_score"], errors="coerce").to_numpy(dtype=float)

            if len(np.unique(y)) == 2 and np.isfinite(s).all():
                return float(average_precision_score(y, s))

        # Some older IF files may use Label
        if "label" in df.columns and "anomaly_score" in df.columns:
            y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
            s = pd.to_numeric(df["anomaly_score"], errors="coerce").to_numpy(dtype=float)

            if len(np.unique(y)) == 2 and np.isfinite(s).all():
                return float(average_precision_score(y, s))

    return np.nan


def load_standard_metrics(path: Path, model_name: str, score_file=None):
    require_file(path)
    df = norm(pd.read_csv(path))

    if df.empty:
        raise ValueError(f"{path.name} ไม่มีข้อมูล")

    row = df.iloc[0]

    result = {
        "model": model_name,
    }

    for m in CORE_METRICS:
        v = numeric_value(row, [m, "f1_score" if m == "f1" else m])
        if v is None:
            raise ValueError(f"{model_name} ขาด metric: {m}")
        result[m] = v

    result["overall_far"] = derive_far(row)

    normal_far = numeric_value(row, ["normal_only_far"])
    result["normal_only_far"] = (
        normal_far if normal_far is not None else result["overall_far"]
    )

    roc_auc = numeric_value(row, ["roc_auc"])
    result["roc_auc"] = roc_auc if roc_auc is not None else np.nan

    result["pr_auc"] = derive_pr_auc(row, score_file)

    test_rows = numeric_value(row, ["test_rows", "final_test_rows", "rows"])
    result["test_rows"] = int(test_rows) if test_rows is not None else np.nan

    for c in ["tn", "fp", "fn", "tp"]:
        v = numeric_value(row, [c])
        result[c] = int(v) if v is not None else np.nan

    return result


def load_if():
    # Preferred IF final metrics
    if IF_METRICS.exists():
        return load_standard_metrics(IF_METRICS, "Isolation Forest", IF_SCORE)

    # Fallback: extract IF row from model_comparison.csv
    require_file(IF_COMBINED)
    df = norm(pd.read_csv(IF_COMBINED))

    if "model" not in df.columns:
        raise ValueError("model_comparison.csv ไม่มี column 'model'")

    mask = df["model"].astype(str).str.strip().str.lower().isin(
        ["isolation forest", "isolationforest", "if"]
    )
    sub = df[mask]

    if sub.empty:
        raise ValueError("ไม่พบ Isolation Forest ใน model_comparison.csv")

    row_df = pd.DataFrame([sub.iloc[0]])
    row_df.to_csv(BASE / "_tmp_if_row.csv", index=False)

    try:
        return load_standard_metrics(
            BASE / "_tmp_if_row.csv",
            "Isolation Forest",
            IF_SCORE
        )
    finally:
        tmp = BASE / "_tmp_if_row.csv"
        if tmp.exists():
            tmp.unlink()


def clean_output(df):
    # Preserve useful ordering for report tables
    cols = [
        "model",
        "test_rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "overall_far",
        "normal_only_far",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return df[cols]


def main():
    print("=" * 100)
    print("STAGE 39 — FINAL THREE-MODEL COMPARISON")
    print("=" * 100)

    rows = [
        load_if(),
        load_standard_metrics(
            OCSVM_METRICS,
            "One-Class SVM"
        ),
        load_standard_metrics(
            LOF_METRICS,
            "Local Outlier Factor"
        ),
    ]

    comparison = clean_output(pd.DataFrame(rows))

    # --------------------------------------------------------
    # Same Final Test integrity check
    # --------------------------------------------------------
    test_rows = comparison["test_rows"].dropna().astype(int).unique()

    if len(test_rows) != 1:
        raise ValueError(
            "โมเดลทั้ง 3 ตัวไม่ได้ใช้ Final Test ชุดเดียวกัน: "
            + ", ".join(str(x) for x in test_rows)
        )

    if test_rows[0] != 9277:
        raise ValueError(
            f"จำนวน Final Test ไม่ตรงกับ CIC pipeline ปัจจุบัน: {test_rows[0]:,} "
            f"(คาดว่า 9,277)"
        )

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.6f"
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------
    lines = [
        "=" * 100,
        "STAGE 39 — FINAL THREE-MODEL COMPARISON",
        "=" * 100,
        "",
        "Final Test was used only for final evaluation/comparison.",
        "No retraining or threshold tuning was performed in Stage 39.",
        "",
    ]

    header = (
        f"{'Model':<22}"
        f"{'Accuracy':>10}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'FAR':>10}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, r in comparison.iterrows():
        def fmt(x):
            return f"{x:.4f}" if pd.notna(x) else "N/A"

        lines.append(
            f"{r['model']:<22}"
            f"{fmt(r['accuracy']):>10}"
            f"{fmt(r['precision']):>11}"
            f"{fmt(r['recall']):>10}"
            f"{fmt(r['f1']):>10}"
            f"{fmt(r['overall_far']):>10}"
            f"{fmt(r['roc_auc']):>10}"
            f"{fmt(r['pr_auc']):>10}"
        )

    lines += ["", "CONFUSION MATRICES", ""]

    for _, r in comparison.iterrows():
        def count_fmt(x):
            return f"{x:,.0f}" if pd.notna(x) else "N/A"

        lines.append(str(r["model"]))
        lines.append(
            f"  TN={count_fmt(r['tn'])}, "
            f"FP={count_fmt(r['fp'])}, "
            f"FN={count_fmt(r['fn'])}, "
            f"TP={count_fmt(r['tp'])}"
        )
        lines.append("")

    lines += [
        "Final Test used only for evaluation: YES",
        "Final Test used for tuning/retraining: NO",
    ]

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "overall_far",
        "roc_auc",
        "pr_auc",
    ]

    plot_df = comparison.set_index("model")[metric_names]

    ax = plot_df.plot(
        kind="bar",
        figsize=(13, 6)
    )
    ax.set_title("Final Test Comparison — IF vs OCSVM vs LOF")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print()
    print(
        f"{'Model':<22}"
        f"{'Accuracy':>10}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'FAR':>10}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
    )
    print("-" * 103)

    for _, r in comparison.iterrows():
        def f4(x):
            return f"{x:.4f}" if pd.notna(x) else "N/A"

        print(
            f"{r['model']:<22}"
            f"{f4(r['accuracy']):>10}"
            f"{f4(r['precision']):>11}"
            f"{f4(r['recall']):>10}"
            f"{f4(r['f1']):>10}"
            f"{f4(r['overall_far']):>10}"
            f"{f4(r['roc_auc']):>10}"
            f"{f4(r['pr_auc']):>10}"
        )

    print()
    print("CONFUSION MATRICES")
    for _, r in comparison.iterrows():
        print(
            f"{r['model']:<22} "
            f"TN={r['tn']:,.0f}  "
            f"FP={r['fp']:,.0f}  "
            f"FN={r['fn']:,.0f}  "
            f"TP={r['tp']:,.0f}"
        )

    print()
    print("Final Test used only for evaluation: YES")
    print("Final Test used for tuning/retraining: NO")
    print(f"Saved CSV     : {OUTPUT_CSV}")
    print(f"Saved Summary : {OUTPUT_TXT}")
    print(f"Saved Chart   : {OUTPUT_PNG}")
    print("=" * 100)


if __name__ == "__main__":
    main()
