from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Validation scores from STAGE 25 using the LOCKED/BEST LOF configuration.
# This is the correct source for Risk/Emotion thresholds.
VALIDATION_SCORE_FILE = BASE_DIR / "lof_best_validation_scores.csv"
SCORE_FILE = BASE_DIR / "final_lof_anomaly_scores.csv"
OUTPUT_FILE = BASE_DIR / "final_lof_risk_emotion.csv"
THRESHOLD_OUT = BASE_DIR / "final_lof_risk_thresholds.json"
ANALYSIS_FILE = BASE_DIR / "final_lof_risk_summary.csv"

for p in [VALIDATION_SCORE_FILE, SCORE_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

validation = pd.read_csv(VALIDATION_SCORE_FILE)
test = pd.read_csv(SCORE_FILE)

required_val = {"anomaly_score", "target"}
required_test = {"anomaly_score", "target"}

missing_val = required_val - set(validation.columns)
missing_test = required_test - set(test.columns)
if missing_val:
    raise ValueError(f"Validation score file ขาดคอลัมน์: {sorted(missing_val)}")
if missing_test:
    raise ValueError(f"Final Test score file ขาดคอลัมน์: {sorted(missing_test)}")

# Risk/Emotion thresholds MUST come from VALIDATION BENIGN scores only.
y_val = pd.to_numeric(validation["target"], errors="coerce").fillna(-1).astype(int)
val_benign = pd.to_numeric(
    validation.loc[y_val == 0, "anomaly_score"], errors="coerce"
).dropna()

if len(val_benign) == 0:
    raise ValueError("ไม่มี VALIDATION BENIGN anomaly score")

p60, p80, p90, p95 = [
    float(np.percentile(val_benign, p)) for p in [60, 80, 90, 95]
]


def classify(v: float) -> int:
    if v <= p60:
        return 1
    if v <= p80:
        return 2
    if v <= p90:
        return 3
    if v <= p95:
        return 4
    return 5


risk_names = {
    1: "Calm",
    2: "Tension",
    3: "Alert",
    4: "High Risk",
    5: "Critical",
}

emotions = {
    1: "Happy",
    2: "Anxious",
    3: "Alert",
    4: "Stressed",
    5: "Angry",
}

# Apply the locked Validation-derived thresholds to Final Test only.
result = test.copy()
result["risk_level"] = [classify(float(v)) for v in result["anomaly_score"]]
result["risk_name"] = result["risk_level"].map(risk_names)
result["network_emotion"] = result["risk_level"].map(emotions)
result.to_csv(OUTPUT_FILE, index=False)

threshold_payload = {
    "risk_p60": p60,
    "risk_p80": p80,
    "risk_p90": p90,
    "risk_p95": p95,
    "source": "VALIDATION BENIGN anomaly scores only",
    "validation_rows": int(len(validation)),
    "validation_benign_rows": int(len(val_benign)),
    "final_test_used_for_threshold": False,
}
THRESHOLD_OUT.write_text(
    json.dumps(threshold_payload, indent=2), encoding="utf-8"
)

rows = []
for level in range(1, 6):
    sub = result[result["risk_level"] == level]
    target = pd.to_numeric(sub["target"], errors="coerce").fillna(-1).astype(int)
    rows.append(
        {
            "risk_level": level,
            "risk_name": risk_names[level],
            "network_emotion": emotions[level],
            "total": len(sub),
            "benign": int((target == 0).sum()),
            "attack": int((target == 1).sum()),
            "percentage": float(len(sub) / len(result) * 100) if len(result) else 0.0,
        }
    )
pd.DataFrame(rows).to_csv(ANALYSIS_FILE, index=False)

print("=" * 80)
print("STAGE 28 — LOF NETWORK RISK / EMOTION")
print("=" * 80)
print(f"Final Test rows : {len(result):,}")
print(f"Validation rows : {len(validation):,}")
print(f"Validation BENIGN used for thresholds: {len(val_benign):,}")
print(f"P60             : {p60:.10f}")
print(f"P80             : {p80:.10f}")
print(f"P90             : {p90:.10f}")
print(f"P95             : {p95:.10f}")
print("Source          : VALIDATION BENIGN only")
print("Final Test used for threshold: NO")
print(f"Saved           : {OUTPUT_FILE}")
print(f"Saved thresholds: {THRESHOLD_OUT}")
print("=" * 80)
