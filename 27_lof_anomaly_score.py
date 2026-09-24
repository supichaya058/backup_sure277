from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR=Path(__file__).resolve().parent
TEST_FILE=BASE_DIR/"test_dataset.csv"
FEATURE_FILE=BASE_DIR/"selected_features.json"
MODEL_FILE=BASE_DIR/"final_lof_model.pkl"
PREDICTION_FILE=BASE_DIR/"final_lof_predictions.csv"
THRESHOLD_FILE=BASE_DIR/"final_lof_threshold.json"
OUTPUT_FILE=BASE_DIR/"final_lof_anomaly_scores.csv"
SUMMARY_FILE=BASE_DIR/"final_lof_anomaly_score_summary.txt"

for p in [TEST_FILE,FEATURE_FILE,MODEL_FILE,PREDICTION_FILE,THRESHOLD_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

features=json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
features=[str(v).strip() for v in features if str(v).strip()!="flow_count"]
test=pd.read_csv(TEST_FILE)
model=joblib.load(MODEL_FILE)
pred=pd.read_csv(PREDICTION_FILE)
cfg=json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
threshold=float(cfg["threshold"])

if len(test)!=len(pred):
    raise ValueError("Final Test/prediction rows ไม่ตรงกัน")

x=test[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
raw=-model.decision_function(x)

result=test.copy()
result["anomaly_score"]=raw
result["detection_threshold"]=threshold
result["prediction"]=pred["prediction"].astype(int).to_numpy()
result["prediction_label"]=result["prediction"].map({0:"Normal",1:"Anomaly"})
result["evaluation_order"]=np.arange(len(result))
result.to_csv(OUTPUT_FILE,index=False)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 FINAL LOF ANOMALY SCORE\n"+"="*70+"\n"
    + f"rows={len(result)}\nthreshold={threshold:.10f}\n"
    + f"min={raw.min():.10f}\nmax={raw.max():.10f}\nmean={raw.mean():.10f}\n"
    + "Final Test used for tuning=False\n",
    encoding="utf-8"
)

print("="*80)
print("STAGE 27 — LOF FINAL TEST ANOMALY SCORE")
print("="*80)
print(f"Final Test rows : {len(result):,}")
print(f"BENIGN          : {(result['target'].astype(int)==0).sum():,}")
print(f"ATTACK          : {(result['target'].astype(int)==1).sum():,}")
print(f"Threshold       : {threshold:.10f}")
print("Threshold source: Validation")
print("Final Test used for tuning: NO")
print(f"Saved           : {OUTPUT_FILE}")
print("="*80)
