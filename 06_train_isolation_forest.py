from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_cic.csv"
TEST_FILE = BASE_DIR / "test_cic.csv"

MODEL_FILE = BASE_DIR / "isolation_forest_cic.joblib"

N_ESTIMATORS = 300
CONTAMINATION = 0.01
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ISOLATION FOREST - BASELINE MODEL")
print("=" * 80)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train rows : {len(train_df):,}")
print(f"Test rows  : {len(test_df):,}")


# ============================================================
# PREPARE X / y
# ============================================================

DROP_COLUMNS = ["target"]

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column not in DROP_COLUMNS
]

X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

y_test = test_df["target"]


print(f"Features   : {len(FEATURE_COLUMNS)}")


# ============================================================
# BUILD MODEL
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "isolation_forest",
        IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 80)
print("TRAINING")
print("=" * 80)

model.fit(X_train)

print("Training complete")


# ============================================================
# PREDICT
# ============================================================

print()
print("=" * 80)
print("TESTING")
print("=" * 80)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

raw_prediction = model.predict(X_test)

prediction = np.where(
    raw_prediction == -1,
    1,
    0
)


# ============================================================
# ANOMALY SCORE
# ============================================================

scores = model.decision_function(X_test)

print()
print("=" * 80)
print("ANOMALY SCORE")
print("=" * 80)

print(f"Min    : {scores.min():.6f}")
print(f"Max    : {scores.max():.6f}")
print(f"Mean   : {scores.mean():.6f}")
print(f"Median : {np.median(scores):.6f}")


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 80)
print("MODEL RESULT")
print("=" * 80)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    prediction
)

print()
print("Confusion Matrix")
print(cm)

print()
print("Format:")
print("[[TN  FP]")
print(" [FN  TP]]")


# ============================================================
# PREDICTION COUNT
# ============================================================

print()
print("=" * 80)
print("PREDICTION COUNT")
print("=" * 80)

print(
    f"Predicted Normal  : {(prediction == 0).sum():,}"
)

print(
    f"Predicted Attack  : {(prediction == 1).sum():,}"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print()
print(f"Model saved to:")
print(MODEL_FILE)

print("=" * 80)
print("COMPLETE")
print("=" * 80)