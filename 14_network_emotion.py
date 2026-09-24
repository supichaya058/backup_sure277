from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# STAGE 14: NETWORK EMOTION / ROLLING RISK
# ============================================================
# Interpretation layer only. It does not tune or retrain.
# ============================================================
BASE=Path(__file__).resolve().parent
SCORE=BASE/'anomaly_scores.csv'; TEST=BASE/'test_behavior.csv'; TRAIN=BASE/'train_behavior.csv'; FEAT=BASE/'selected_features.json'; OUT=BASE/'emotion_results.csv'; COMP=BASE/'final_behavior_risk_emotion.csv'; TH=BASE/'emotion_thresholds.json'; SUMMARY=BASE/'emotion_engine_summary.txt'
ROLLING_WINDOW=50
RISK_NAMES={1:'Calm',2:'Tension',3:'Alert',4:'High Risk',5:'Critical'}
EMOTION_MAP={1:'Happy',2:'Anxious',3:'Alert',4:'Stressed',5:'Angry'}

def rolling_percentile(s,w):
    x=pd.to_numeric(s,errors='coerce').fillna(0).to_numpy(float); r=np.zeros(len(x))
    for i,v in enumerate(x):
        cur=x[max(0,i-w+1):i+1]; r[i]=float(np.mean(cur<=v))
    return r

def risk(v):
    if v<.2:return 1
    if v<.4:return 2
    if v<.6:return 3
    if v<.8:return 4
    return 5
for p in [SCORE,TEST,TRAIN,FEAT]:
    if not p.exists(): raise FileNotFoundError(f'ไม่พบไฟล์: {p}')
score=pd.read_csv(SCORE); test=pd.read_csv(TEST); train=pd.read_csv(TRAIN); fs=json.loads(FEAT.read_text(encoding='utf-8')); fs=[f for f in dict.fromkeys(str(x).strip() for x in fs) if f!='flow_count']
if len(score)!=len(test): raise ValueError('Score/Test row mismatch')
r=score.sort_values('evaluation_order').reset_index(drop=True) if 'evaluation_order' in score.columns else score.copy()
r['rolling_percentile']=rolling_percentile(r['anomaly_score'],ROLLING_WINDOW); r['risk_level']=r['rolling_percentile'].map(risk); r['risk_name']=r['risk_level'].map(RISK_NAMES); r['network_emotion']=r['risk_level'].map(EMOTION_MAP)
tr=train[fs].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0); mu=tr.mean(); sd=tr.std().replace(0,np.nan).fillna(1)
tx=r[fs].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0); z=((tx-mu)/sd).abs(); r['dominant_feature']=z.idxmax(axis=1); r['dominant_feature_zscore']=z.max(axis=1)
r.to_csv(OUT,index=False); r.to_csv(COMP,index=False)
TH.write_text(json.dumps({'rolling_window':ROLLING_WINDOW,'score_reference':'TRAIN BENIGN','risk_boundaries':{'risk_1':'<0.20','risk_2':'>=0.20 and <0.40','risk_3':'>=0.40 and <0.60','risk_4':'>=0.60 and <0.80','risk_5':'>=0.80'},'risk_to_emotion':EMOTION_MAP},indent=2,ensure_ascii=False),encoding='utf-8')
SUMMARY.write_text('CIC-IDS2017 NETWORK EMOTION\n'+'='*60+'\n' + f'Test rows={len(r):,}\nScore reference=TRAIN BENIGN\nRolling window={ROLLING_WINDOW}\n',encoding='utf-8')
print('='*80); print('STAGE 14 — NETWORK EMOTION'); print('='*80); print('Final Test rows:',len(r)); print('Interpretation only; no tuning.'); print('Saved:',OUT)
