from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

BASE_DIR=Path(__file__).resolve().parent
SCORE_FILE=BASE_DIR/"final_lof_anomaly_scores.csv"
RISK_FILE=BASE_DIR/"final_lof_risk_emotion.csv"
METRICS_FILE=BASE_DIR/"final_lof_final_metrics.csv"
ATTACK_FILE=BASE_DIR/"final_lof_attack_analysis.csv"
SUMMARY_FILE=BASE_DIR/"final_lof_analysis_summary.txt"

for p in [SCORE_FILE,RISK_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

score=pd.read_csv(SCORE_FILE)
risk=pd.read_csv(RISK_FILE)
if len(score)!=len(risk):
    raise ValueError("LOF score/risk rows ไม่ตรงกัน")

result=score.copy()
for c in ["risk_level","risk_name","network_emotion"]:
    if c in risk.columns:
        result[c]=risk[c].to_numpy()

y=result["target"].astype(int).to_numpy()
pred=result["prediction"].astype(int).to_numpy()
scores=result["anomaly_score"].astype(float).to_numpy()
tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()

metrics={
    "model":"Local Outlier Factor",
    "test_rows":int(len(result)),
    "accuracy":float(accuracy_score(y,pred)),
    "precision":float(precision_score(y,pred,zero_division=0)),
    "recall":float(recall_score(y,pred,zero_division=0)),
    "f1":float(f1_score(y,pred,zero_division=0)),
    "overall_far":float(fp/(fp+tn) if fp+tn else 0.0),
    "normal_only_far":float(fp/(fp+tn) if fp+tn else 0.0),
    "roc_auc":float(roc_auc_score(y,scores)),
    "pr_auc":float(average_precision_score(y,scores)),
    "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
}
pd.DataFrame([metrics]).to_csv(METRICS_FILE,index=False)

if "attack_type" in result.columns:
    attack=result[result["target"].astype(int)==1].groupby("attack_type",dropna=False).agg(
        samples=("target","size"),
        detected=("prediction","sum"),
        mean_score=("anomaly_score","mean"),
        median_score=("anomaly_score","median"),
    ).reset_index()
    attack["detection_rate"]=attack["detected"]/attack["samples"]
else:
    attack=pd.DataFrame()
attack.to_csv(ATTACK_FILE,index=False)

print("="*90)
print("STAGE 29 — FINAL LOF TEST ANALYSIS")
print("="*90)
print(f"Final Test rows : {len(result):,}")
print()
print(f"{'Metric':<22}{'Value':>12}")
print("-"*34)
for k in ["accuracy","precision","recall","f1","overall_far","normal_only_far","roc_auc","pr_auc"]:
    print(f"{k:<22}{metrics[k]:>12.4f}")
print()
print("CONFUSION MATRIX")
print("                 Pred Normal   Pred Anomaly")
print(f"Actual BENIGN      {tn:>10,}   {fp:>12,}")
print(f"Actual ATTACK      {fn:>10,}   {tp:>12,}")
print()
print("Final Test used only for evaluation: YES")
print("Final Test used for tuning        : NO")
print(f"Saved metrics : {METRICS_FILE}")
print(f"Saved attack  : {ATTACK_FILE}")
print("="*90)

SUMMARY_FILE.write_text(
    "CIC-IDS2017 LOF FINAL TEST ANALYSIS\n"+"="*70+"\n"
    + "\n".join(f"{k}={v}" for k,v in metrics.items())+"\n"
    + "Final Test used only for evaluation=YES\nFinal Test used for tuning=NO\n",
    encoding="utf-8"
)
