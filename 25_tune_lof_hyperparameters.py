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
THRESHOLD_CONFIG = BASE_DIR / "lof_threshold_config.json"
RESULT_FILE = BASE_DIR / "lof_hyperparameter_tuning_results.csv"
BEST_CONFIG_FILE = BASE_DIR / "lof_best_config.json"
BEST_MODEL_FILE = BASE_DIR / "lof_behavior_best_validation.pkl"
BEST_SCORE_FILE = BASE_DIR / "lof_best_validation_scores.csv"
SUMMARY_FILE = BASE_DIR / "lof_hyperparameter_tuning_summary.txt"

N_NEIGHBORS_VALUES = [10,20,30,50]
CONTAMINATION_VALUES = [0.005,0.01,0.03,0.05]
MAX_FAR = 0.15

for p in [TRAIN_FILE,VALIDATION_FILE,FEATURE_FILE,THRESHOLD_CONFIG]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

train=pd.read_csv(TRAIN_FILE)
val=pd.read_csv(VALIDATION_FILE)
features=json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
features=[str(v).strip() for v in features if str(v).strip()!="flow_count"]
cfg=json.loads(THRESHOLD_CONFIG.read_text(encoding="utf-8"))
pct=float(cfg["threshold_percentile"])

if (train["target"].astype(int)!=0).any():
    raise ValueError("Train ต้องมี BENIGN only")
if "source_file" in train.columns and "source_file" in val.columns:
    if set(train["source_file"].astype(str)) & set(val["source_file"].astype(str)):
        raise ValueError("พบ group overlap Train/Validation")

x_train=train[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
x_val=val[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
y=val["target"].astype(int).to_numpy()
benign_mask=y==0
rows=[]

for n_neighbors in N_NEIGHBORS_VALUES:
    for contamination in CONTAMINATION_VALUES:
        model=Pipeline([
            ("scaler",StandardScaler()),
            ("lof",LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                metric="minkowski",p=2,novelty=True,n_jobs=-1
            ))
        ])
        model.fit(x_train)
        scores=-model.decision_function(x_val)
        threshold=float(np.percentile(scores[benign_mask],pct))
        pred=(scores>=threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        fpr=fp/(fp+tn) if fp+tn else 0.0
        rows.append({
            "n_neighbors":n_neighbors,
            "contamination":contamination,
            "threshold_percentile":pct,
            "threshold":threshold,
            "accuracy":accuracy_score(y,pred),
            "precision":precision_score(y,pred,zero_division=0),
            "recall":recall_score(y,pred,zero_division=0),
            "f1":f1_score(y,pred,zero_division=0),
            "fpr":fpr,
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "eligible_far_le_15pct":bool(fpr<=MAX_FAR),
        })

results=pd.DataFrame(rows)
eligible=results[results["eligible_far_le_15pct"]].copy()
if eligible.empty:
    raise RuntimeError("ไม่มี LOF configuration ที่ผ่าน Normal-only FAR <= 15%")

best_row=eligible.sort_values(
    ["f1","recall","fpr","precision"],
    ascending=[False,False,True,False]
).iloc[0]

best_model=Pipeline([
    ("scaler",StandardScaler()),
    ("lof",LocalOutlierFactor(
        n_neighbors=int(best_row["n_neighbors"]),
        contamination=float(best_row["contamination"]),
        metric="minkowski",p=2,novelty=True,n_jobs=-1
    ))
])
best_model.fit(x_train)
best_scores=-best_model.decision_function(x_val)
joblib.dump(best_model,BEST_MODEL_FILE)

best_pred=(best_scores>=float(best_row["threshold"])).astype(int)
best_score_out=val.copy()
best_score_out["anomaly_score"]=best_scores
best_score_out["detection_threshold"]=float(best_row["threshold"])
best_score_out["prediction"]=best_pred
best_score_out["prediction_label"]=pd.Series(best_pred).map({0:"Normal",1:"Anomaly"})
best_score_out.to_csv(BEST_SCORE_FILE,index=False)

best_cfg={
    "n_neighbors":int(best_row["n_neighbors"]),
    "contamination":float(best_row["contamination"]),
    "metric":"minkowski","p":2,"novelty":True,"n_jobs":-1,
    "threshold_percentile":pct,
    "threshold":float(best_row["threshold"]),
    "validation_fpr":float(best_row["fpr"]),
    "validation_f1":float(best_row["f1"]),
    "selection_constraint":"Normal-only FAR <= 15%",
    "final_test_used_for_tuning":False,
}
BEST_CONFIG_FILE.write_text(json.dumps(best_cfg,indent=2),encoding="utf-8")
results.to_csv(RESULT_FILE,index=False)

print("="*90)
print("STAGE 25 — LOF HYPERPARAMETER TUNING (VALIDATION ONLY)")
print("="*90)
print(f"Threshold percentile : {pct:.0f}")
print(f"Eligible configs     : {len(eligible)} / {len(results)}")
print()
print(f"Best n_neighbors     : {best_cfg['n_neighbors']}")
print(f"Best contamination   : {best_cfg['contamination']}")
print(f"Threshold            : {best_cfg['threshold']:.10f}")
print(f"Validation Precision : {best_cfg['validation_f1'] and best_row['precision']:.4f}")
print(f"Validation Recall    : {best_row['recall']:.4f}")
print(f"Validation F1        : {best_cfg['validation_f1']:.4f}")
print(f"Validation FAR       : {best_cfg['validation_fpr']:.4f}")
print("FAR constraint       : PASS")
print("Final Test used      : NO")
print(f"Saved config         : {BEST_CONFIG_FILE}")
print("="*90)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 LOF HYPERPARAMETER TUNING\n"+"="*70+"\n"
    + "\n".join(f"{k}={v}" for k,v in best_cfg.items())+"\n",
    encoding="utf-8"
)
