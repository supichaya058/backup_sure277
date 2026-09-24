from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

# ============================================================
# STAGE 13: FINAL ISOLATION FOREST ANOMALY SCORE ON FINAL TEST
# ============================================================
# No score ranking is calculated inside Final Test.
# Raw scores are kept, and normalized scores are empirical percentiles
# against TRAIN BENIGN scores only.
# ============================================================
BASE=Path(__file__).resolve().parent
TEST=BASE/'test_behavior.csv'; TRAIN=BASE/'train_behavior.csv'; FEAT=BASE/'selected_features.json'; MODEL=BASE/'final_anomaly_model.pkl'; OUT=BASE/'anomaly_scores.csv'; TRAINOUT=BASE/'train_anomaly_scores.csv'

def clean(df,fs): return df[fs].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(dtype=float)
def ref_pct(values,ref):
    ref=np.sort(np.asarray(ref,dtype=float)); vals=np.asarray(values,dtype=float)
    return np.searchsorted(ref,vals,side='right')/len(ref)

for p in [TEST,TRAIN,FEAT,MODEL]:
    if not p.exists(): raise FileNotFoundError(f'ไม่พบไฟล์: {p}')
test=pd.read_csv(TEST); train=pd.read_csv(TRAIN); fs=json.loads(FEAT.read_text(encoding='utf-8')); fs=[f for f in dict.fromkeys(str(x).strip() for x in fs) if f!='flow_count']
ben=train[train['target'].astype(int)==0].copy(); model=joblib.load(MODEL)
tr=clean(ben,fs); te=clean(test,fs); raw_tr=-model.decision_function(tr); raw_te=-model.decision_function(te)
out=test.copy(); out['anomaly_score_raw']=raw_te; out['anomaly_score']=ref_pct(raw_te,raw_tr); out['evaluation_order']=np.arange(len(out)); out.to_csv(OUT,index=False)
trainout=ben.copy(); trainout['anomaly_score_raw']=raw_tr; trainout['anomaly_score']=ref_pct(raw_tr,raw_tr); trainout.to_csv(TRAINOUT,index=False)
print('='*80); print('STAGE 13 — IF FINAL TEST SCORE'); print('='*80); print('Final Test:',len(out)); print('Score calibration: TRAIN BENIGN only'); print('Final Test used to calibrate score: NO'); print('Saved:',OUT)
