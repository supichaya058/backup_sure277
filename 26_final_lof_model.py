from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR=Path(__file__).resolve().parent
TRAIN_FILE=BASE_DIR/"train_behavior.csv"
TEST_FILE=BASE_DIR/"test_dataset.csv"
FEATURE_FILE=BASE_DIR/"selected_features.json"
CONFIG_FILE=BASE_DIR/"lof_best_config.json"
MODEL_FILE=BASE_DIR/"final_lof_model.pkl"
PREDICTION_FILE=BASE_DIR/"final_lof_predictions.csv"
THRESHOLD_FILE=BASE_DIR/"final_lof_threshold.json"
TRAIN_SCORE_FILE=BASE_DIR/"final_lof_train_scores.csv"
SUMMARY_FILE=BASE_DIR/"final_lof_model_summary.txt"

for p in [TRAIN_FILE,TEST_FILE,FEATURE_FILE,CONFIG_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

train=pd.read_csv(TRAIN_FILE)
test=pd.read_csv(TEST_FILE)
features=json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
features=[str(v).strip() for v in features if str(v).strip()!="flow_count"]
cfg=json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

if (train["target"].astype(int)!=0).any():
    raise ValueError("Final LOF training ต้องเป็น BENIGN only")

n_neighbors=int(cfg["n_neighbors"])
contamination=float(cfg["contamination"])
threshold=float(cfg["threshold"])

model=Pipeline([
    ("scaler",StandardScaler()),
    ("lof",LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric=cfg.get("metric","minkowski"),
        p=int(cfg.get("p",2)),
        novelty=True,
        n_jobs=int(cfg.get("n_jobs",-1))
    ))
])
x_train=train[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
x_test=test[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
model.fit(x_train)
joblib.dump(model,MODEL_FILE)

train_score=-model.decision_function(x_train)
test_score=-model.decision_function(x_test)
train_out=train.copy()
train_out["anomaly_score"]=train_score
train_out.to_csv(TRAIN_SCORE_FILE,index=False)

test_out=test.copy()
test_out["anomaly_score"]=test_score
test_out["detection_threshold"]=threshold
test_out["prediction"]=(test_score>=threshold).astype(int)
test_out["prediction_label"]=test_out["prediction"].map({0:"Normal",1:"Anomaly"})
test_out.to_csv(PREDICTION_FILE,index=False)

THRESHOLD_FILE.write_text(json.dumps({
    "threshold":threshold,
    "threshold_percentile":cfg["threshold_percentile"],
    "source":"Validation-selected threshold; final model trained on TRAIN BENIGN only",
    "final_test_used_for_threshold":False
},indent=2),encoding="utf-8")

print("="*90)
print("STAGE 26 — FINAL LOF MODEL")
print("="*90)
print(f"Train BENIGN      : {len(train):,}")
print(f"Final Test        : {len(test):,}")
print(f"Features          : {len(features)}")
print()
print("LOCKED CONFIGURATION")
print(f"n_neighbors       : {n_neighbors}")
print(f"contamination     : {contamination}")
print(f"threshold pct     : {cfg['threshold_percentile']}")
print(f"threshold         : {threshold:.10f}")
print()
print("Final Test used for training  : NO")
print("Final Test used for threshold : NO")
print(f"Model saved       : {MODEL_FILE}")
print(f"Predictions saved : {PREDICTION_FILE}")
print("="*90)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 FINAL LOF MODEL\n"+"="*70+"\n"
    + f"n_neighbors={n_neighbors}\ncontamination={contamination}\n"
    + f"threshold_percentile={cfg['threshold_percentile']}\nthreshold={threshold}\n"
    + "Final Test used for training=False\nFinal Test used for threshold=False\n",
    encoding="utf-8"
)
