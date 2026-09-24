from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# STAGE 19: FINAL ONE-CLASS SVM MODEL
# ============================================================
# Hyperparameters + threshold are locked from Validation.
# Final model is trained on TRAIN BENIGN only.
# Final Test is not used for training or threshold calculation.
# ============================================================

BASE = Path(__file__).resolve().parent

TRAIN = BASE / "train_behavior.csv"
VALIDATION = BASE / "validation_behavior.csv"
TEST = BASE / "test_dataset.csv"
FEAT = BASE / "selected_features.json"
CFG = BASE / "one_class_svm_best_config.json"

MODEL = BASE / "final_one_class_svm_behavior.pkl"
PRED = BASE / "final_one_class_svm_predictions.csv"
TH = BASE / "final_one_class_svm_threshold.json"
DEVOUT = BASE / "final_one_class_svm_train_scores.csv"
SUMMARY = BASE / "final_one_class_svm_model_summary.txt"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")


def load_features() -> list[str]:
    with open(FEAT, "r", encoding="utf-8") as f:
        features = json.load(f)

    if not isinstance(features, list) or not features:
        raise ValueError("selected_features.json ต้องเป็น list และต้องมี feature")

    features = list(dict.fromkeys(str(v).strip() for v in features))
    features = [v for v in features if v and v != "flow_count"]
    return features


def clean_x(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            "Data ไม่มี feature ที่จำเป็น:\n" + "\n".join(missing)
        )

    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )


def parse_gamma(value):
    """
    รองรับทั้ง 'scale', 'auto', และค่าตัวเลขที่อาจถูกบันทึกเป็น string เช่น '1.0'.
    """
    if isinstance(value, str):
        value = value.strip()
        if value in {"scale", "auto"}:
            return value
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"gamma ใน one_class_svm_best_config.json ไม่ถูกต้อง: {value!r}"
            ) from exc

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    raise ValueError(f"gamma ใน config ไม่รองรับ: {value!r}")


def validate_split_integrity(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, df in [("Train", train), ("Validation", validation), ("Final Test", test)]:
        if "source_file" not in df.columns:
            continue

        groups = set(df["source_file"].astype(str))
        if name == "Train":
            train_groups = groups
        elif name == "Validation":
            overlap = groups & train_groups
            if overlap:
                raise ValueError(
                    "พบ GROUP LEAKAGE ระหว่าง Train และ Validation: "
                    + ", ".join(sorted(overlap))
                )
        else:
            overlap_train = groups & train_groups
            overlap_val = groups & set(validation["source_file"].astype(str))
            if overlap_train or overlap_val:
                raise ValueError(
                    "พบ GROUP LEAKAGE ใน Final Test\n"
                    f"Train overlap: {sorted(overlap_train)}\n"
                    f"Validation overlap: {sorted(overlap_val)}"
                )

    # Final Test ต้องไม่เป็นสำเนาเดียวกับ Validation
    if len(validation) == len(test):
        try:
            if validation.equals(test):
                raise ValueError(
                    "พบ DATA LEAKAGE: validation_behavior.csv เหมือน test_dataset.csv"
                )
        except TypeError:
            pass


def main() -> None:
    for path in [TRAIN, VALIDATION, TEST, FEAT, CFG]:
        require_file(path)

    train = pd.read_csv(TRAIN)
    validation = pd.read_csv(VALIDATION)
    test = pd.read_csv(TEST)
    features = load_features()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))

    if "target" not in train.columns:
        raise ValueError("train_behavior.csv ต้องมี column target")

    if int(pd.to_numeric(train["target"], errors="coerce").fillna(0).sum()) != 0:
        raise ValueError("Train ต้องมี BENIGN เท่านั้น")

    validate_split_integrity(train, validation, test)

    # --------------------------------------------------------
    # Locked configuration from Stage 18
    # --------------------------------------------------------
    kernel = cfg.get("kernel", "rbf")
    nu = float(cfg["nu"])
    gamma = parse_gamma(cfg.get("gamma", "scale"))
    threshold = float(cfg["threshold"])
    threshold_percentile = cfg.get("threshold_percentile", None)

    # Stage 18 writes the best model based on Validation.
    # The numeric threshold stored in CFG is LOCKED and is not recalculated here.
    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ocsvm",
            OneClassSVM(
                kernel=kernel,
                nu=nu,
                gamma=gamma,
                cache_size=4096,
                shrinking=True,
            ),
        ),
    ])

    x_train = clean_x(train, features)
    x_test = clean_x(test, features)

    # --------------------------------------------------------
    # TRAIN FINAL MODEL ON TRAIN BENIGN ONLY
    # --------------------------------------------------------
    model.fit(x_train)
    joblib.dump(model, MODEL)

    # --------------------------------------------------------
    # SCORE TRAIN + FINAL TEST
    # --------------------------------------------------------
    train_score = -model.decision_function(x_train)
    test_score = -model.decision_function(x_test)

    train_out = train.copy()
    train_out["anomaly_score"] = train_score
    train_out.to_csv(DEVOUT, index=False)

    test_out = test.copy()
    test_out["anomaly_score"] = test_score
    test_out["prediction"] = (test_score >= threshold).astype(int)
    test_out["prediction_label"] = test_out["prediction"].map({0: "Normal", 1: "Anomaly"})
    test_out["detection_threshold"] = threshold
    test_out.to_csv(PRED, index=False)

    TH.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "threshold_percentile": threshold_percentile,
                "source": "Validation-locked threshold from Stage 18",
                "final_test_used_for_threshold": False,
                "final_model_training_data": "TRAIN BENIGN only",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    SUMMARY.write_text(
        "CIC-IDS2017 FINAL ONE-CLASS SVM\n"
        + "=" * 70
        + "\n"
        + f"train_benign_rows={len(train):,}\n"
        + f"validation_rows={len(validation):,}\n"
        + f"final_test_rows={len(test):,}\n"
        + f"features={len(features)}\n"
        + f"kernel={kernel}\n"
        + f"nu={nu}\n"
        + f"gamma={gamma}\n"
        + f"threshold_percentile={threshold_percentile}\n"
        + f"threshold={threshold}\n"
        + "final_test_used_for_threshold=False\n"
        + "final_model_training_data=TRAIN BENIGN only\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------
    print("=" * 85)
    print("STAGE 19 — FINAL ONE-CLASS SVM MODEL")
    print("=" * 85)
    print()
    print("[DATA]")
    print(f"  Train BENIGN      : {len(train):,}")
    print(f"  Validation        : {len(validation):,}")
    print(f"  Final Test        : {len(test):,}")
    print(f"  Features          : {len(features)}")
    print()
    print("[LOCKED CONFIGURATION — FROM VALIDATION]")
    print(f"  kernel            : {kernel}")
    print(f"  nu                : {nu}")
    print(f"  gamma             : {gamma}")
    print(f"  threshold pct     : {threshold_percentile}")
    print(f"  threshold         : {threshold:.8f}")
    print()
    print("[FINAL MODEL]")
    print("  Training data     : TRAIN BENIGN only")
    print("  Final Test used for training   : NO")
    print("  Final Test used for threshold  : NO")
    print("  Model saved       :", MODEL)
    print()
    print("[FINAL TEST SCORING READY]")
    print(f"  Test scores       : {len(test_score):,}")
    print(f"  Predicted Normal  : {(test_out['prediction'] == 0).sum():,}")
    print(f"  Predicted Anomaly : {(test_out['prediction'] == 1).sum():,}")
    print()
    print("[SAVED FILES]")
    print(" ", PRED)
    print(" ", TH)
    print(" ", DEVOUT)
    print(" ", SUMMARY)
    print("=" * 85)
    print("STAGE 19 FINAL ONE-CLASS SVM COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
