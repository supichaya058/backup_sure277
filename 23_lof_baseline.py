from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train_behavior.csv"
VALIDATION_FILE = BASE_DIR / "validation_behavior.csv"
FEATURE_FILE = BASE_DIR / "selected_features.json"
MODEL_FILE = BASE_DIR / "lof_behavior_baseline.pkl"
SCORE_FILE = BASE_DIR / "lof_baseline_validation_scores.csv"
SUMMARY_FILE = BASE_DIR / "lof_baseline_summary.txt"

N_NEIGHBORS = 20
CONTAMINATION = 0.05
METRIC = "minkowski"
P = 2
N_JOBS = -1

for p in [TRAIN_FILE, VALIDATION_FILE, FEATURE_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VALIDATION_FILE)
features = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
features = [str(v).strip() for v in features if str(v).strip() != "flow_count"]

if (train["target"].astype(int) != 0).any():
    raise ValueError("Train LOF ต้องมี BENIGN only")
if set(train["source_file"].astype(str)) & set(val["source_file"].astype(str)):
    raise ValueError("พบ group overlap Train/Validation")

x_train = train[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
x_val = val[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
y = val["target"].astype(int).to_numpy()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lof", LocalOutlierFactor(
        n_neighbors=N_NEIGHBORS,
        contamination=CONTAMINATION,
        metric=METRIC,
        p=P,
        novelty=True,
        n_jobs=N_JOBS,
    )),
])
model.fit(x_train)
joblib.dump(model, MODEL_FILE)

score = -model.decision_function(x_val)
pred = (score >= 0.0).astype(int)
cm = confusion_matrix(y,pred,labels=[0,1])
tn,fp,fn,tp = cm.ravel()

result = val.copy()
result["anomaly_score"] = score
result["prediction"] = pred
result["prediction_label"] = result["prediction"].map({0:"Normal",1:"Anomaly"})
result.to_csv(SCORE_FILE,index=False)

print("="*80)
print("STAGE 23 — LOF BASELINE (VALIDATION ONLY)")
print("="*80)
print(f"Train rows       : {len(train):,}")
print(f"Validation rows  : {len(val):,}")
print(f"Features         : {len(features)}")
print(f"Train BENIGN     : {(train['target'].astype(int)==0).sum():,}")
print(f"Validation BENIGN: {(y==0).sum():,}")
print(f"Validation ATTACK: {(y==1).sum():,}")
print()
print("PARAMETERS")
print(f"n_neighbors : {N_NEIGHBORS}")
print(f"contamination: {CONTAMINATION}")
print(f"metric      : {METRIC}")
print(f"novelty     : True")
print(f"native threshold: 0.0")
print()
print("VALIDATION RESULT")
print(f"Accuracy  : {accuracy_score(y,pred):.4f}")
print(f"Precision : {precision_score(y,pred,zero_division=0):.4f}")
print(f"Recall    : {recall_score(y,pred,zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y,pred,zero_division=0):.4f}")
print(f"FPR       : {fp/(fp+tn) if fp+tn else 0.0:.4f}")
print()
print(f"Model saved : {MODEL_FILE}")
print(f"Scores saved: {SCORE_FILE}")
print("Final Test used: NO")
print("="*80)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 LOF BASELINE\n" + "="*70 + "\n"
    + f"n_neighbors={N_NEIGHBORS}\ncontamination={CONTAMINATION}\n"
    + f"accuracy={accuracy_score(y,pred):.6f}\nprecision={precision_score(y,pred,zero_division=0):.6f}\n"
    + f"recall={recall_score(y,pred,zero_division=0):.6f}\nf1={f1_score(y,pred,zero_division=0):.6f}\n"
    + f"fpr={fp/(fp+tn) if fp+tn else 0.0:.6f}\nFinal Test used: NO\n", encoding="utf-8"
)
