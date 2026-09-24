from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
SCORE_FILE = BASE_DIR / "lof_baseline_validation_scores.csv"
OUTPUT_FILE = BASE_DIR / "lof_threshold_tuning_results.csv"
CONFIG_FILE = BASE_DIR / "lof_threshold_config.json"
SUMMARY_FILE = BASE_DIR / "lof_threshold_summary.txt"

PERCENTILES = [20,30,40,50,60,70,75,80,85,90,92,95,97,99]
MAX_FAR = 0.15

if not SCORE_FILE.exists():
    raise FileNotFoundError(f"ไม่พบไฟล์: {SCORE_FILE}")

df = pd.read_csv(SCORE_FILE)
y = df["target"].astype(int).to_numpy()
score = df["anomaly_score"].astype(float).to_numpy()
benign = score[y == 0]

def calc(th):
    pred = (score >= th).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    far = fp/(fp+tn) if fp+tn else 0.0
    return {
        "threshold": float(th),
        "accuracy": accuracy_score(y,pred),
        "precision": precision_score(y,pred,zero_division=0),
        "recall": recall_score(y,pred,zero_division=0),
        "f1": f1_score(y,pred,zero_division=0),
        "fpr": far, "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }

rows=[]
# Native threshold
rows.append({"type":"native","percentile":0.0,**calc(0.0)})
for p in PERCENTILES:
    th=float(np.percentile(benign,p))
    rows.append({"type":"percentile","percentile":float(p),**calc(th)})

result=pd.DataFrame(rows)
eligible=result[result["fpr"] <= MAX_FAR].copy()
if eligible.empty:
    raise RuntimeError("ไม่มี LOF threshold ที่ผ่าน Normal-only FAR <= 15%")

best=eligible.sort_values(
    ["f1","recall","fpr","precision"],
    ascending=[False,False,True,False]
).iloc[0]

cfg={
    "threshold_percentile": float(best["percentile"]),
    "threshold": float(best["threshold"]),
    "selection_constraint":"Normal-only FAR <= 15%",
    "validation_fpr":float(best["fpr"]),
    "validation_f1":float(best["f1"]),
    "final_test_used_for_tuning":False,
}
result["eligible_far_le_15pct"] = result["fpr"] <= MAX_FAR
result.to_csv(OUTPUT_FILE,index=False)
CONFIG_FILE.write_text(json.dumps(cfg,indent=2),encoding="utf-8")

print("="*85)
print("STAGE 24 — LOF THRESHOLD TUNING (VALIDATION ONLY)")
print("="*85)
print(f"Eligible thresholds : {len(eligible)} / {len(result)}")
print(f"Best percentile     : {cfg['threshold_percentile']:.0f}")
print(f"Best threshold      : {cfg['threshold']:.10f}")
print(f"Validation F1       : {cfg['validation_f1']:.4f}")
print(f"Normal-only FAR     : {cfg['validation_fpr']:.4f}")
print("FAR constraint <=15%: PASS")
print("Final Test used     : NO")
print(f"Saved results       : {OUTPUT_FILE}")
print(f"Saved config        : {CONFIG_FILE}")
print("="*85)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 LOF THRESHOLD TUNING\n"+"="*70+"\n"
    + f"best_percentile={cfg['threshold_percentile']}\n"
    + f"best_threshold={cfg['threshold']}\n"
    + f"validation_f1={cfg['validation_f1']}\n"
    + f"validation_fpr={cfg['validation_fpr']}\n"
    + "constraint=Normal-only FAR <= 15%\nFinal Test used=NO\n",
    encoding="utf-8"
)
