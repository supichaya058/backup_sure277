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
# STAGE 11: IF MAX_SAMPLES / MAX_FEATURES TUNING
# ============================================================
# Validation only. The Final Test is not loaded.
# Uses the best n_estimators selected in Stage 10.
# ============================================================

BASE_DIR=Path(__file__).resolve().parent
TRAIN=BASE_DIR/'train_behavior.csv'; VAL=BASE_DIR/'validation_behavior.csv'
FEAT=BASE_DIR/'selected_features.json'; ST10=BASE_DIR/'if_stage10_config.json'
OUT=BASE_DIR/'behavior_samples_features_results.csv'; BEST=BASE_DIR/'behavior_best_samples_features.txt'
MODEL=BASE_DIR/'behavior_best_samples_features_model.pkl'; CFG=BASE_DIR/'behavior_tuning_config.json'

MAX_SAMPLES_VALUES=['auto',0.5,0.8]
MAX_FEATURES_VALUES=[1.0,0.8]
CONTAMINATION=0.01; RANDOM_STATE=42; THRESHOLD_PERCENTILE=95

def clean(df,fs): return df[fs].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(dtype=np.float64)
def metrics(y,s,th):
    p=(s>=th).astype(int); cm=confusion_matrix(y,p,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return {'accuracy':accuracy_score(y,p),'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),'fpr':fp/(fp+tn) if fp+tn else 0.0,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

train=pd.read_csv(TRAIN); val=pd.read_csv(VAL); fs=json.loads(FEAT.read_text(encoding='utf-8')); fs=[f for f in dict.fromkeys(str(x).strip() for x in fs) if f!='flow_count']
cfg10=json.loads(ST10.read_text(encoding='utf-8')); n=int(cfg10['n_estimators'])
xtr=clean(train,fs); xv=clean(val,fs); y=val['target'].astype(int).to_numpy(); benign=y==0
rows=[]; best=None
for ms in MAX_SAMPLES_VALUES:
  for mf in MAX_FEATURES_VALUES:
    model=Pipeline([('scaler',StandardScaler()),('iso',IsolationForest(n_estimators=n,contamination=CONTAMINATION,max_samples=ms,max_features=mf,random_state=RANDOM_STATE,n_jobs=-1))])
    model.fit(xtr); score=-model.decision_function(xv); th=float(np.percentile(score[benign],THRESHOLD_PERCENTILE)); m=metrics(y,score,th)
    row={'n_estimators':n,'max_samples':ms,'max_features':mf,'contamination':CONTAMINATION,'threshold_percentile':THRESHOLD_PERCENTILE,'threshold':th,**m}; rows.append(row)
    if best is None or (row['f1'],-row['fpr'],row['precision'])>(best['f1'],-best['fpr'],best['precision']): best={**row,'model':model}

pd.DataFrame(rows).to_csv(OUT,index=False); joblib.dump(best['model'],MODEL)
BEST.write_text(f"n_estimators={n}\nmax_samples={best['max_samples']}\nmax_features={best['max_features']}\n",encoding='utf-8')
config={k:v for k,v in best.items() if k!='model'}; config['selection_data']='VALIDATION'; config['final_test_used_for_tuning']=False; CFG.write_text(json.dumps(config,indent=2),encoding='utf-8')
print('='*90); print('STAGE 11 — IF MAX_SAMPLES / MAX_FEATURES TUNING (VALIDATION ONLY)'); print('='*90); print(pd.DataFrame(rows).to_string(index=False)); print('\nBest:',json.dumps(config,indent=2)); print('Final Test used for tuning: NO')
