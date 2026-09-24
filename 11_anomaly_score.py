import pandas as pd
import numpy as np
import joblib


# =========================
# 1. โหลดข้อมูลและโมเดล
# =========================

test = pd.read_csv("test_cic.csv")

model = joblib.load(
    "final_isolation_forest.pkl"
)

scaler = joblib.load(
    "final_scaler.pkl"
)


# =========================
# 2. เตรียมข้อมูล
# =========================

X_test = test.drop(
    columns=["target", "attack_type"]
)


# =========================
# 3. ปรับขนาดข้อมูล
# =========================

X_test_scaled = scaler.transform(
    X_test
)


# =========================
# 4. หา Anomaly Score
# =========================

score = model.decision_function(
    X_test_scaled
)


# =========================
# 5. เก็บผล
# =========================

result = test.copy()

result["anomaly_score"] = score


# =========================
# 6. แสดงผล
# =========================

print("=" * 70)
print("ANOMALY SCORE")
print("=" * 70)

print(
    f"จำนวนข้อมูล : {len(result):,}"
)

print(
    f"Min    : {score.min():.6f}"
)

print(
    f"Max    : {score.max():.6f}"
)

print(
    f"Mean   : {score.mean():.6f}"
)

print(
    f"Median : {np.median(score):.6f}"
)


# =========================
# 7. แสดงตัวอย่าง
# =========================

print()
print("=" * 70)
print("ตัวอย่างข้อมูล")
print("=" * 70)

print(
    result[
        [
            "attack_type",
            "target",
            "anomaly_score"
        ]
    ].head(10).to_string(index=False)
)


# =========================
# 8. ดูคะแนนแยกตามประเภท
# =========================

print()
print("=" * 70)
print("ANOMALY SCORE ตามประเภท")
print("=" * 70)

summary = (
    result
    .groupby("attack_type")["anomaly_score"]
    .agg(
        Count="count",
        Mean="mean",
        Min="min",
        Max="max",
        Median="median"
    )
    .sort_values("Mean")
)


print(summary.to_string())


# =========================
# 9. บันทึกผล
# =========================

result.to_csv(
    "anomaly_scores.csv",
    index=False
)

summary.to_csv(
    "anomaly_score_summary.csv"
)


# =========================
# 10. เสร็จ
# =========================

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print("บันทึกไฟล์:")
print("- anomaly_scores.csv")
print("- anomaly_score_summary.csv")

print("=" * 70)