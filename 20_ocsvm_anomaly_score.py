from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "test_dataset.csv"
FEATURE_FILE = BASE_DIR / "selected_features.json"
MODEL_FILE = BASE_DIR / "final_one_class_svm_behavior.pkl"
PREDICTION_FILE = BASE_DIR / "final_one_class_svm_predictions.csv"
THRESHOLD_FILE = BASE_DIR / "final_one_class_svm_threshold.json"
OUTPUT_FILE = BASE_DIR / "final_one_class_svm_anomaly_scores.csv"
SUMMARY_FILE = BASE_DIR / "final_one_class_svm_anomaly_score_summary.txt"

for p in [TEST_FILE, FEATURE_FILE, MODEL_FILE, PREDICTION_FILE, THRESHOLD_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

features = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
features = [str(v).strip() for v in features if str(v).strip() != "flow_count"]

test = pd.read_csv(TEST_FILE)
model = joblib.load(MODEL_FILE)
pred_df = pd.read_csv(PREDICTION_FILE)
cfg = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))

if len(test) != len(pred_df):
    raise ValueError("Final Test กับ prediction rows ไม่ตรงกัน")
if "prediction" not in pred_df.columns:
    raise ValueError("Prediction file ไม่มี column prediction")

threshold = float(cfg.get("threshold", cfg.get("final_threshold")))
x_test = (
    test[features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .to_numpy(dtype=np.float64)
)
raw = -model.decision_function(x_test)

result = test.copy()
result["anomaly_score"] = raw
result["detection_threshold"] = threshold
result["prediction"] = pred_df["prediction"].astype(int).to_numpy()
result["prediction_label"] = result["prediction"].map({0: "Normal", 1: "Anomaly"})
result["evaluation_order"] = np.arange(len(result))
result.to_csv(OUTPUT_FILE, index=False)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 FINAL ONE-CLASS SVM ANOMALY SCORE\n"
    + "=" * 70 + "\n"
    + f"Final Test rows = {len(result)}\n"
    + f"BENIGN = {(result['target'].astype(int) == 0).sum()}\n"
    + f"ATTACK = {(result['target'].astype(int) == 1).sum()}\n"
    + f"threshold = {threshold:.10f}\n"
    + f"min_score = {raw.min():.10f}\n"
    + f"max_score = {raw.max():.10f}\n"
    + f"mean_score = {raw.mean():.10f}\n"
    + "Final Test used for tuning = NO\n",
    encoding="utf-8",
)

print("=" * 80)
print("STAGE 20 — OCSVM FINAL TEST ANOMALY SCORE")
print("=" * 80)
print(f"Test rows       : {len(result):,}")
print(f"Actual BENIGN   : {(result['target'].astype(int) == 0).sum():,}")
print(f"Actual ATTACK   : {(result['target'].astype(int) == 1).sum():,}")
print(f"Features        : {len(features)}")
print(f"Threshold       : {threshold:.10f}")
print("Threshold source: Validation")
print("Final Test used for tuning: NO")
print(f"Saved           : {OUTPUT_FILE}")
print("=" * 80)
