from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# STAGE 10: ISOLATION FOREST N_ESTIMATORS TUNING
# ============================================================
# Validation only. Final Test is NOT loaded.
# Threshold is calibrated from Validation BENIGN scores for each candidate.
# Selection rule: F1 first, then lower FPR, then higher precision.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TRAIN = BASE_DIR / "train_behavior.csv"
VAL = BASE_DIR / "validation_behavior.csv"
FEATURES = BASE_DIR / "selected_features.json"
OUT = BASE_DIR / "behavior_n_estimators_results.csv"
BEST = BASE_DIR / "behavior_best_n_estimators.txt"
MODEL = BASE_DIR / "behavior_best_n_estimators_model.pkl"
CFG = BASE_DIR / "if_stage10_config.json"

N_ESTIMATORS_VALUES = [100, 200, 300, 500, 700, 1000]
CONTAMINATION = 0.01
RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 95


def clean(df, fs):
    return df[fs].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)


def metrics(y, score, th):
    pred = (score >= th).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': accuracy_score(y,pred), 'precision': precision_score(y,pred,zero_division=0),
        'recall': recall_score(y,pred,zero_division=0), 'f1': f1_score(y,pred,zero_division=0),
        'fpr': fp/(fp+tn) if fp+tn else 0.0, 'tn': int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)
    }

train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)
fs = json.loads(FEATURES.read_text(encoding='utf-8'))
fs = [f for f in dict.fromkeys(str(x).strip() for x in fs) if f != 'flow_count']
if int(train['target'].sum()) != 0:
    raise ValueError('Train must be BENIGN only')

xtr = clean(train,fs); xv = clean(val,fs); y=val['target'].astype(int).to_numpy(); benign=y==0
rows=[]; best=None
for n in N_ESTIMATORS_VALUES:
    model = Pipeline([('scaler',StandardScaler()),('iso',IsolationForest(n_estimators=n, contamination=CONTAMINATION, random_state=RANDOM_STATE,n_jobs=-1))])
    model.fit(xtr)
    score=-model.decision_function(xv)
    th=float(np.percentile(score[benign], THRESHOLD_PERCENTILE))
    m=metrics(y,score,th)
    row={'n_estimators':n,'contamination':CONTAMINATION,'threshold_percentile':THRESHOLD_PERCENTILE,'threshold':th,**m}
    rows.append(row)
    if best is None or (row['f1'],-row['fpr'],row['precision'])>(best['f1'],-best['fpr'],best['precision']):
        best={**row,'model':model}

pd.DataFrame(rows).to_csv(OUT,index=False)
joblib.dump(best['model'],MODEL)
BEST.write_text(str(best['n_estimators']),encoding='utf-8')
CFG.write_text(json.dumps({k:v for k,v in best.items() if k!='model'},indent=2),encoding='utf-8')

print('='*90); print('STAGE 10 — IF N_ESTIMATORS TUNING (VALIDATION ONLY)'); print('='*90)
print(pd.DataFrame(rows).to_string(index=False))
print('\nBest:',best['n_estimators'],'threshold=',f"{best['threshold']:.8f}",'F1=',f"{best['f1']:.4f}")
print('Final Test used for tuning: NO')
