import pandas as pd


# =========================
# 1. โหลดผลจากขั้น 12
# =========================

data = pd.read_csv(
    "network_emotion_result.csv"
)


# =========================
# 2. กำหนดลำดับ Emotion
# =========================

emotion_order = [
    "Happy",
    "Alert",
    "Anxious",
    "Stressed",
    "Angry"
]


# =========================
# 3. ผลรวมทั้งหมด
# =========================

print("=" * 80)
print("NETWORK EMOTION ANALYSIS")
print("=" * 80)

print(
    f"ข้อมูลทั้งหมด : {len(data):,} rows"
)


# =========================
# 4. Emotion Distribution
# =========================

print()
print("=" * 80)
print("OVERALL EMOTION")
print("=" * 80)

emotion_count = (
    data["emotion"]
    .value_counts()
    .reindex(
        emotion_order,
        fill_value=0
    )
)

for emotion, count in emotion_count.items():

    percentage = (
        count / len(data)
    ) * 100

    print(
        f"{emotion:<12}"
        f"{count:>10,} "
        f"({percentage:>6.2f}%)"
    )


# =========================
# 5. Risk Distribution
# =========================

print()
print("=" * 80)
print("OVERALL RISK LEVEL")
print("=" * 80)

risk_count = (
    data["risk_level"]
    .value_counts()
    .sort_index()
)

for risk, count in risk_count.items():

    percentage = (
        count / len(data)
    ) * 100

    emotion = (
        data.loc[
            data["risk_level"] == risk,
            "emotion"
        ]
        .mode()
        .iloc[0]
    )

    print(
        f"Risk {risk} "
        f"{emotion:<10}"
        f"{count:>10,} "
        f"({percentage:>6.2f}%)"
    )


# =========================
# 6. วิเคราะห์แต่ละ Attack
# =========================

print()
print("=" * 80)
print("ATTACK TYPE SUMMARY")
print("=" * 80)

attack_types = (
    data["attack_type"]
    .unique()
)


summary = []


for attack_type in attack_types:

    group = data[
        data["attack_type"] == attack_type
    ]

    # จำนวนข้อมูล
    count = len(group)

    # ค่าเฉลี่ยคะแนน
    mean_score = (
        group["anomaly_score"]
        .mean()
    )

    # ค่ากลางคะแนน
    median_score = (
        group["anomaly_score"]
        .median()
    )

    # Risk ที่พบมากที่สุด
    dominant_risk = (
        group["risk_level"]
        .mode()
        .iloc[0]
    )

    # Emotion ที่พบมากที่สุด
    dominant_emotion = (
        group["emotion"]
        .mode()
        .iloc[0]
    )

    # จำนวน Emotion แต่ละแบบ
    emotion_counts = (
        group["emotion"]
        .value_counts()
    )

    # เปอร์เซ็นต์ Emotion ที่เด่นที่สุด
    dominant_emotion_percent = (
        emotion_counts.get(
            dominant_emotion,
            0
        )
        / count
        * 100
    )

    summary.append({
        "Attack Type": attack_type,
        "Count": count,
        "Mean Score": mean_score,
        "Median Score": median_score,
        "Main Risk": dominant_risk,
        "Main Emotion": dominant_emotion,
        "Main Emotion %": dominant_emotion_percent
    })


summary_df = pd.DataFrame(summary)


# =========================
# 7. เรียงตามคะแนน
# =========================

summary_df = summary_df.sort_values(
    by="Mean Score"
)


# =========================
# 8. แสดงผล
# =========================

print()

print(
    f"{'Attack Type':<35}"
    f"{'Count':>10}"
    f"{'Mean Score':>14}"
    f"{'Main Risk':>12}"
    f"{'Main Emotion':>15}"
    f"{'Emotion %':>12}"
)

print("-" * 100)


for _, row in summary_df.iterrows():

    print(
        f"{str(row['Attack Type']):<35}"
        f"{int(row['Count']):>10,}"
        f"{row['Mean Score']:>14.6f}"
        f"{int(row['Main Risk']):>12}"
        f"{str(row['Main Emotion']):>15}"
        f"{row['Main Emotion %']:>11.2f}%"
    )


# =========================
# 9. Emotion ของแต่ละ Attack
# =========================

print()
print("=" * 80)
print("EMOTION BY ATTACK TYPE")
print("=" * 80)

emotion_table = pd.crosstab(
    data["attack_type"],
    data["emotion"],
    normalize="index"
) * 100

emotion_table = emotion_table.reindex(
    columns=emotion_order,
    fill_value=0
)

print(
    emotion_table.round(2).to_string()
)


# =========================
# 10. Risk ของแต่ละ Attack
# =========================

print()
print("=" * 80)
print("RISK BY ATTACK TYPE")
print("=" * 80)

risk_table = pd.crosstab(
    data["attack_type"],
    data["risk_level"],
    normalize="index"
) * 100

risk_table = risk_table.reindex(
    columns=[1, 2, 3, 4, 5],
    fill_value=0
)

risk_table.columns = [
    "Risk 1",
    "Risk 2",
    "Risk 3",
    "Risk 4",
    "Risk 5"
]

print(
    risk_table.round(2).to_string()
)


# =========================
# 11. บันทึกผล
# =========================

summary_df.to_csv(
    "attack_summary.csv",
    index=False
)

emotion_table.to_csv(
    "emotion_by_attack.csv"
)

risk_table.to_csv(
    "risk_by_attack.csv"
)


# =========================
# 12. เสร็จ
# =========================

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print("สร้างไฟล์:")
print("- attack_summary.csv")
print("- emotion_by_attack.csv")
print("- risk_by_attack.csv")

print("=" * 80)