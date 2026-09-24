from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# STAGE 12: FINAL ISOLATION FOREST MODEL
# ============================================================
# Final model is trained on TRAIN BENIGN only.
# Hyperparameters and threshold were locked from Validation.
# Final Test is not loaded here.
#
# This stage also evaluates the finalized model on VALIDATION only
# using the already-locked threshold. This is a verification/reporting
# result; it does not tune any parameter and does not use Final Test.
# ============================================================

BASE = Path(__file__).resolve().parent
TRAIN = BASE / 'train_behavior.csv'
VALIDATION = BASE / 'validation_behavior.csv'
FEAT = BASE / 'selected_features.json'
CFG = BASE / 'behavior_tuning_config.json'
MODEL = BASE / 'final_anomaly_model.pkl'
LEGACY = BASE / 'final_behavior_isolation_forest.pkl'
OUTCFG = BASE / 'final_if_config.json'
SUMMARY = BASE / 'final_model_summary.txt'
VALIDATION_SCORES = BASE / 'final_if_validation_scores.csv'
VALIDATION_METRICS = BASE / 'final_if_validation_metrics.csv'

train = pd.read_csv(TRAIN)
validation = pd.read_csv(VALIDATION)
fs = json.loads(FEAT.read_text(encoding='utf-8'))
fs = [f for f in dict.fromkeys(str(x).strip() for x in fs) if f != 'flow_count']
cfg = json.loads(CFG.read_text(encoding='utf-8'))

if int(train['target'].sum()) != 0:
    raise ValueError('Train must contain BENIGN only')

params = {
    'n_estimators': int(cfg['n_estimators']),
    'contamination': float(cfg.get('contamination', 0.01)),
    'max_samples': cfg.get('max_samples', 'auto'),
    'max_features': float(cfg.get('max_features', 1.0)),
    'random_state': 42,
    'n_jobs': -1,
}

threshold = float(cfg['threshold'])

x_train = (
    train[fs]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .to_numpy(dtype=np.float64)
)

x_validation = (
    validation[fs]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .to_numpy(dtype=np.float64)
)

y_validation = validation['target'].astype(int).to_numpy()

model = Pipeline([
    ('scaler', StandardScaler()),
    ('iso', IsolationForest(**params)),
])

# ------------------------------------------------------------
# FINAL TRAINING — TRAIN ONLY
# ------------------------------------------------------------
model.fit(x_train)
joblib.dump(model, MODEL)
joblib.dump(model, LEGACY)

# ------------------------------------------------------------
# VALIDATION VERIFICATION — LOCKED THRESHOLD ONLY
# ------------------------------------------------------------
raw_score = -model.decision_function(x_validation)
prediction = (raw_score >= threshold).astype(int)

accuracy = accuracy_score(y_validation, prediction)
precision = precision_score(y_validation, prediction, zero_division=0)
recall = recall_score(y_validation, prediction, zero_division=0)
f1 = f1_score(y_validation, prediction, zero_division=0)

cm = confusion_matrix(y_validation, prediction, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) else 0.0

validation_result = validation.copy()
validation_result['anomaly_score'] = raw_score
validation_result['detection_threshold'] = threshold
validation_result['prediction'] = prediction
validation_result['prediction_label'] = pd.Series(prediction).map({0: 'Normal', 1: 'Anomaly'}).to_numpy()
validation_result.to_csv(VALIDATION_SCORES, index=False)

metrics_row = {
    'dataset': 'CIC-IDS2017',
    'split': 'Validation',
    'rows': len(validation),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'fpr': fpr,
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'tp': int(tp),
    'n_estimators': params['n_estimators'],
    'contamination': params['contamination'],
    'max_samples': params['max_samples'],
    'max_features': params['max_features'],
    'threshold_percentile': cfg['threshold_percentile'],
    'threshold': threshold,
    'final_test_used': False,
}
pd.DataFrame([metrics_row]).to_csv(VALIDATION_METRICS, index=False)

OUTCFG.write_text(
    json.dumps({
        'features': fs,
        'model_params': params,
        'threshold_percentile': cfg['threshold_percentile'],
        'validation_threshold': threshold,
        'final_test_used_for_tuning': False,
        'validation_verification': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        },
    }, indent=2),
    encoding='utf-8'
)

summary_text = (
    'CIC-IDS2017 FINAL ISOLATION FOREST MODEL\n'
    + '=' * 70 + '\n'
    + f'Train BENIGN rows = {len(train):,}\n'
    + f'Validation rows = {len(validation):,}\n'
    + f'n_estimators = {params["n_estimators"]}\n'
    + f'contamination = {params["contamination"]}\n'
    + f'max_samples = {params["max_samples"]}\n'
    + f'max_features = {params["max_features"]}\n'
    + f'threshold_percentile = {cfg["threshold_percentile"]}\n'
    + f'threshold = {threshold:.12f}\n'
    + '\n'
    + 'VALIDATION VERIFICATION (LOCKED CONFIGURATION)\n'
    + '=' * 70 + '\n'
    + f'Accuracy  = {accuracy:.6f}\n'
    + f'Precision = {precision:.6f}\n'
    + f'Recall    = {recall:.6f}\n'
    + f'F1        = {f1:.6f}\n'
    + f'FPR       = {fpr:.6f}\n'
    + f'TN        = {tn}\n'
    + f'FP        = {fp}\n'
    + f'FN        = {fn}\n'
    + f'TP        = {tp}\n'
    + '\n'
    + f'Final Test used = NO\n'
    + f'Validation scores saved = {VALIDATION_SCORES.name}\n'
    + f'Validation metrics saved = {VALIDATION_METRICS.name}\n'
)
SUMMARY.write_text(summary_text, encoding='utf-8')

print('=' * 85)
print('STAGE 12 — FINAL ISOLATION FOREST')
print('=' * 85)
print('Train BENIGN:', len(train))
print('n_estimators:', params['n_estimators'])
print('max_samples:', params['max_samples'])
print('max_features:', params['max_features'])
print('Validation threshold:', threshold)
print('Final Test used: NO')
print()
print('=' * 85)
print('FINAL MODEL VALIDATION VERIFICATION')
print('=' * 85)
print(f'Validation rows : {len(validation):,}')
print(f'Accuracy        : {accuracy:.4f}')
print(f'Precision       : {precision:.4f}')
print(f'Recall          : {recall:.4f}')
print(f'F1 Score        : {f1:.4f}')
print(f'FPR             : {fpr:.4f}')
print()
print('Confusion Matrix')
print(cm)
print(f'TN : {tn:,}')
print(f'FP : {fp:,}')
print(f'FN : {fn:,}')
print(f'TP : {tp:,}')
print()
print(f'Saved validation scores  : {VALIDATION_SCORES.name}')
print(f'Saved validation metrics : {VALIDATION_METRICS.name}')
print(f'Saved final model        : {MODEL.name}')
print('=' * 85)
print('STAGE 12 FINAL ISOLATION FOREST COMPLETE')
print('=' * 85)
