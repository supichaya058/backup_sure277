import pandas as pd
import numpy as np
import joblib


# =========================
# 1. โหลดข้อมูล
# =========================

train = pd.read_csv("train_cic.csv")
result = pd.read_csv("anomaly_scores.csv")

model = joblib.load(
    "final_isolation_forest.pkl"
)

scaler = joblib.load(
    "final_scaler.pkl"
)


# =========================
# 2. เตรียมข้อมูล Train
# =========================

X_train = train.drop(
    columns=["target"]
)

X_train_scaled = scaler.transform(
    X_train
)


# =========================
# 3. หา Score ของข้อมูลปกติ
# =========================

normal_scores = model.decision_function(
    X_train_scaled
)


# =========================
# 4. สร้างจุดแบ่ง Risk Level
# =========================

p60 = np.percentile(
    normal_scores,
    60
)

p40 = np.percentile(
    normal_scores,
    40
)

p20 = np.percentile(
    normal_scores,
    20
)

p5 = np.percentile(
    normal_scores,
    5
)


print("=" * 70)
print("RISK LEVEL")
print("=" * 70)

print(f"P60 = {p60:.6f}")
print(f"P40 = {p40:.6f}")
print(f"P20 = {p20:.6f}")
print(f"P05 = {p5:.6f}")


# =========================
# 5. กำหนด Risk Level
# =========================

def get_risk(score):

    # คะแนนสูง = ปกติมากกว่า
    if score >= p60:
        return 1

    elif score >= p40:
        return 2

    elif score >= p20:
        return 3

    elif score >= p5:
        return 4

    else:
        return 5


result["risk_level"] = (
    result["anomaly_score"]
    .apply(get_risk)
)


# =========================
# 6. แปลงเป็น Emotion
# =========================

emotion_map = {
    1: "Happy",
    2: "Alert",
    3: "Anxious",
    4: "Stressed",
    5: "Angry"
}

result["emotion"] = (
    result["risk_level"]
    .map(emotion_map)
)


# =========================
# 7. แสดงจำนวนแต่ละ Risk
# =========================

print()
print("=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

risk_count = (
    result["risk_level"]
    .value_counts()
    .sort_index()
)

for risk, count in risk_count.items():

    emotion = emotion_map[risk]

    percentage = (
        count / len(result)
    ) * 100

    print(
        f"Risk {risk} "
        f"{emotion:<10} : "
        f"{count:>8,} "
        f"({percentage:>6.2f}%)"
    )


# =========================
# 8. ดูคะแนนเฉลี่ยตาม Risk
# =========================

print()
print("=" * 70)
print("AVERAGE SCORE BY RISK")
print("=" * 70)

risk_summary = (
    result
    .groupby(
        ["risk_level", "emotion"]
    )["anomaly_score"]
    .agg(
        Count="count",
        Mean="mean",
        Min="min",
        Max="max"
    )
    .reset_index()
)

print(
    risk_summary.to_string(
        index=False
    )
)


# =========================
# 9. ดู Risk ของแต่ละประเภท
# =========================

print()
print("=" * 70)
print("RISK BY ATTACK TYPE")
print("=" * 70)

attack_risk = (
    result
    .groupby(
        ["attack_type", "risk_level", "emotion"]
    )
    .size()
    .reset_index(
        name="Count"
    )
)

print(
    attack_risk.to_string(
        index=False
    )
)


# =========================
# 10. ดูตัวอย่างข้อมูล
# =========================

print()
print("=" * 70)
print("SAMPLE RESULT")
print("=" * 70)

print(
    result[
        [
            "attack_type",
            "target",
            "anomaly_score",
            "risk_level",
            "emotion"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# =========================
# 11. บันทึกผล
# =========================

result.to_csv(
    "network_emotion_result.csv",
    index=False
)

risk_summary.to_csv(
    "risk_summary.csv",
    index=False
)

attack_risk.to_csv(
    "risk_by_attack_type.csv",
    index=False
)


# =========================
# 12. เสร็จ
# =========================

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print("สร้างไฟล์:")
print("- network_emotion_result.csv")
print("- risk_summary.csv")
print("- risk_by_attack_type.csv")

print("=" * 70)