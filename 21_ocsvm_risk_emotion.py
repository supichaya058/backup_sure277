from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRAIN_SCORE_FILE = BASE_DIR / "final_one_class_svm_train_scores.csv"
SCORE_FILE = BASE_DIR / "final_one_class_svm_anomaly_scores.csv"
THRESHOLD_FILE = BASE_DIR / "final_one_class_svm_threshold.json"
OUTPUT_FILE = BASE_DIR / "final_one_class_svm_risk_emotion.csv"
THRESHOLD_OUT = BASE_DIR / "final_one_class_svm_risk_thresholds.json"
ANALYSIS_FILE = BASE_DIR / "final_one_class_svm_risk_summary.csv"

for p in [TRAIN_SCORE_FILE, SCORE_FILE, THRESHOLD_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

train_scores = pd.read_csv(TRAIN_SCORE_FILE)
scores = pd.read_csv(SCORE_FILE)
model_cfg = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))

if "anomaly_score" not in train_scores.columns or "anomaly_score" not in scores.columns:
    raise ValueError("Score files ต้องมี anomaly_score")

# Risk thresholds are calibrated from TRAIN BENIGN only.
benign_scores = pd.to_numeric(train_scores["anomaly_score"], errors="coerce").dropna()
if len(benign_scores) == 0:
    raise ValueError("ไม่มี TRAIN BENIGN anomaly score")

p60, p80, p90, p95 = [
    float(np.percentile(benign_scores, p)) for p in [60, 80, 90, 95]
]

def risk_level(v):
    if v <= p60: return 1
    if v <= p80: return 2
    if v <= p90: return 3
    if v <= p95: return 4
    return 5

risk_names = {1:"Calm", 2:"Tension", 3:"Alert", 4:"High Risk", 5:"Critical"}
emotions = {1:"Happy", 2:"Anxious", 3:"Alert", 4:"Stressed", 5:"Angry"}

result = scores.copy()
result["risk_level"] = [risk_level(float(v)) for v in result["anomaly_score"]]
result["risk_name"] = result["risk_level"].map(risk_names)
result["network_emotion"] = result["risk_level"].map(emotions)
result.to_csv(OUTPUT_FILE, index=False)

thresholds = {
    "risk_p60": p60, "risk_p80": p80, "risk_p90": p90, "risk_p95": p95,
    "source": "TRAIN BENIGN anomaly scores only",
    "detection_threshold": float(model_cfg.get("threshold", model_cfg.get("final_threshold"))),
}
THRESHOLD_OUT.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

rows=[]
for level in range(1,6):
    sub=result[result["risk_level"]==level]
    rows.append({
        "risk_level": level,
        "risk_name": risk_names[level],
        "network_emotion": emotions[level],
        "total": len(sub),
        "benign": int((sub["target"].astype(int)==0).sum()) if "target" in sub.columns else None,
        "attack": int((sub["target"].astype(int)==1).sum()) if "target" in sub.columns else None,
        "percentage": float(len(sub)/len(result)*100) if len(result) else 0.0,
    })
pd.DataFrame(rows).to_csv(ANALYSIS_FILE, index=False)

print("=" * 80)
print("STAGE 21 — OCSVM NETWORK RISK / EMOTION")
print("=" * 80)
print(f"Final Test rows : {len(result):,}")
print(f"Risk P60        : {p60:.10f}")
print(f"Risk P80        : {p80:.10f}")
print(f"Risk P90        : {p90:.10f}")
print(f"Risk P95        : {p95:.10f}")
print("Risk source     : TRAIN BENIGN only")
print(f"Saved           : {OUTPUT_FILE}")
print(f"Saved thresholds: {THRESHOLD_OUT}")
print("=" * 80)
