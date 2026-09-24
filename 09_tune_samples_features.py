import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split


# =========================
# 1. โหลดข้อมูล
# =========================

train = pd.read_csv("train_cic.csv")
test = pd.read_csv("test_cic.csv")

X_train = train.drop(columns=["target"])

X_test = test.drop(
    columns=["target", "attack_type"]
)

y_test = test["target"]


# =========================
# 2. แบ่ง Validation / Final Test
# =========================

X_val, X_final, y_val, y_final = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=42,
    stratify=y_test
)

print("=" * 70)
print("TUNE MAX_SAMPLES / MAX_FEATURES")
print("=" * 70)

print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Final Test : {len(X_final):,}")


# =========================
# 3. ปรับขนาดข้อมูล
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_final = scaler.transform(X_final)


# =========================
# 4. ค่าที่ใช้
# =========================

n_estimators = 100
contamination = 0.50

max_samples_values = [
    0.5,
    0.75,
    1.0
]

max_features_values = [
    0.5,
    0.75,
    1.0
]


# =========================
# 5. ทดลอง
# =========================

results = []

print()
print("=" * 70)
print("กำลังทดลอง...")
print("=" * 70)

for max_samples in max_samples_values:

    for max_features in max_features_values:

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train)

        prediction = model.predict(X_val)

        prediction = np.where(
            prediction == -1,
            1,
            0
        )

        precision = precision_score(
            y_val,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            prediction,
            zero_division=0
        )

        results.append({
            "max_samples": max_samples,
            "max_features": max_features,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })


# =========================
# 6. แสดงผล
# =========================

result_df = pd.DataFrame(results)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"{'max_samples':<15}"
    f"{'max_features':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1':<15}"
)

print("-" * 75)

for _, row in result_df.iterrows():

    print(
        f"{row['max_samples']:<15.2f}"
        f"{row['max_features']:<15.2f}"
        f"{row['precision']:<15.4f}"
        f"{row['recall']:<15.4f}"
        f"{row['f1']:<15.4f}"
    )


# =========================
# 7. หาค่าที่ดีที่สุด
# =========================

best = result_df.loc[
    result_df["f1"].idxmax()
]

best_max_samples = best["max_samples"]
best_max_features = best["max_features"]


# =========================
# 8. แสดงค่าที่ดีที่สุด
# =========================

print()
print("=" * 70)
print("BEST RESULT")
print("=" * 70)

print(f"n_estimators : {n_estimators}")
print(f"contamination: {contamination:.2f}")
print(f"max_samples  : {best_max_samples:.2f}")
print(f"max_features : {best_max_features:.2f}")
print(f"Precision    : {best['precision']:.4f}")
print(f"Recall       : {best['recall']:.4f}")
print(f"F1           : {best['f1']:.4f}")


# =========================
# 9. สร้าง Model ด้วยค่าที่ดีที่สุด
# =========================

final_model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    max_samples=best_max_samples,
    max_features=best_max_features,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train)


# =========================
# 10. ทดสอบ Final Test
# =========================

final_prediction = final_model.predict(X_final)

final_prediction = np.where(
    final_prediction == -1,
    1,
    0
)


# =========================
# 11. คำนวณผล
# =========================

final_precision = precision_score(
    y_final,
    final_prediction,
    zero_division=0
)

final_recall = recall_score(
    y_final,
    final_prediction,
    zero_division=0
)

final_f1 = f1_score(
    y_final,
    final_prediction,
    zero_division=0
)


# =========================
# 12. Final Test
# =========================

print()
print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(f"n_estimators : {n_estimators}")
print(f"contamination: {contamination:.2f}")
print(f"max_samples  : {best_max_samples:.2f}")
print(f"max_features : {best_max_features:.2f}")
print(f"Precision    : {final_precision:.4f}")
print(f"Recall       : {final_recall:.4f}")
print(f"F1-score     : {final_f1:.4f}")

print("=" * 70)