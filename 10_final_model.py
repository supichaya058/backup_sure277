import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# =========================
# 1. โหลดข้อมูล
# =========================

train = pd.read_csv("train_cic.csv")
test = pd.read_csv("test_cic.csv")


# =========================
# 2. แยกข้อมูล
# =========================

X_train = train.drop(
    columns=["target"]
)

# แบ่ง Test เหมือนตอนทดลองค่า
test_validation, test_final = train_test_split(
    test,
    test_size=0.5,
    random_state=42,
    stratify=test["target"]
)

X_final = test_final.drop(
    columns=["target", "attack_type"]
)

y_final = test_final["target"]


# =========================
# 3. แสดงข้อมูล
# =========================

print("=" * 70)
print("FINAL ISOLATION FOREST MODEL")
print("=" * 70)

print(f"Train      : {len(X_train):,}")
print(f"Final Test : {len(X_final):,}")
print(f"Features   : {len(X_train.columns)}")


# =========================
# 4. ปรับขนาดข้อมูล
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_final = scaler.transform(X_final)


# =========================
# 5. สร้างโมเดล
# =========================

model = IsolationForest(
    n_estimators=100,
    contamination=0.50,
    max_samples=1.0,
    max_features=0.50,
    random_state=42,
    n_jobs=-1
)


# =========================
# 6. Train
# =========================

print()
print("กำลัง Train...")

model.fit(X_train)

print("Train เสร็จแล้ว")


# =========================
# 7. ทำนาย
# =========================

prediction = model.predict(X_final)

# Isolation Forest
# -1 = ผิดปกติ
#  1 = ปกติ

prediction = np.where(
    prediction == -1,
    1,
    0
)


# =========================
# 8. คำนวณผล
# =========================

precision = precision_score(
    y_final,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_final,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_final,
    prediction,
    zero_division=0
)


# =========================
# 9. แสดงผลรวม
# =========================

print()
print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# =========================
# 10. Confusion Matrix
# =========================

cm = confusion_matrix(
    y_final,
    prediction
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print()
print("[[TN  FP]")
print(" [FN  TP]]")


# =========================
# 11. จำนวนที่โมเดลทำนาย
# =========================

print()
print("=" * 70)
print("PREDICTION COUNT")
print("=" * 70)

print(
    f"ทายว่า Normal : "
    f"{(prediction == 0).sum():,}"
)

print(
    f"ทายว่า Attack : "
    f"{(prediction == 1).sum():,}"
)


# =========================
# 12. ผลแยกตามประเภท Attack
# =========================

result = test_final.copy()

result["prediction"] = prediction


print()
print("=" * 70)
print("ATTACK TYPE RESULT")
print("=" * 70)

attack_result = []

for attack_type, group in result.groupby("attack_type"):

    actual_attack = (
        group["target"] == 1
    ).sum()

    detected_attack = (
        group["prediction"] == 1
    ).sum()

    if actual_attack > 0:

        detection_rate = (
            detected_attack / actual_attack
        )

    else:

        # สำหรับ BENIGN
        actual_normal = len(group)

        false_alarm_rate = (
            detected_attack / actual_normal
        )

        detection_rate = false_alarm_rate

    attack_result.append({
        "Attack Type": attack_type,
        "Actual": len(group),
        "Detected as Attack": detected_attack,
        "Detection Rate": detection_rate
    })


attack_df = pd.DataFrame(
    attack_result
)


for _, row in attack_df.iterrows():

    print(
        f"{row['Attack Type']:<35}"
        f"Actual: {int(row['Actual']):>7,}    "
        f"Detected: {int(row['Detected as Attack']):>7,}    "
        f"Rate: {row['Detection Rate'] * 100:>6.2f}%"
    )


# =========================
# 13. Anomaly Score
# =========================

score = model.decision_function(
    X_final
)

result["anomaly_score"] = score


print()
print("=" * 70)
print("ANOMALY SCORE")
print("=" * 70)

print(f"Min    : {score.min():.6f}")
print(f"Max    : {score.max():.6f}")
print(f"Mean   : {score.mean():.6f}")
print(f"Median : {np.median(score):.6f}")


# =========================
# 14. บันทึกผล
# =========================

result.to_csv(
    "final_predictions.csv",
    index=False
)

attack_df.to_csv(
    "attack_type_result.csv",
    index=False
)

joblib.dump(
    model,
    "final_isolation_forest.pkl"
)

joblib.dump(
    scaler,
    "final_scaler.pkl"
)


# =========================
# 15. เสร็จ
# =========================

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print("บันทึก:")
print("- final_predictions.csv")
print("- attack_type_result.csv")
print("- final_isolation_forest.pkl")
print("- final_scaler.pkl")

print("=" * 70)