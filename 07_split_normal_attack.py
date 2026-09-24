from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd

# ============================================================
# CIC-IDS2017 -> STAGE 07: CLEAN TRAIN / VALIDATION / FINAL TEST SPLIT
# ============================================================
# Comparison protocol:
#   - Train       : BENIGN only
#   - Validation  : BENIGN + ATTACK, used for model/threshold tuning
#   - Final Test  : BENIGN + ATTACK, untouched until final evaluation
#
# Grouping:
#   CIC behavior windows are grouped by source_file (original PCAP).
#   A source_file must never appear in more than one split.
#   This is the closest CIC equivalent to the run/PCAP separation used
#   by the custom dataset and avoids row-level leakage from the same PCAP.
#
# Target split proportions are approximately:
#   Train ~60% / Validation ~20% / Final Test ~20%.
#   Validation is kept large enough to support the clean Hybrid IF + RF
#   RF-train/tune group split because CIC has only a small number of PCAP groups.
#   Exact row counts depend on the available source_file groups.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "behavior_dataset.csv"
TRAIN_FILE = BASE_DIR / "train_behavior.csv"
VALIDATION_FILE = BASE_DIR / "validation_behavior.csv"
TEST_NORMAL_FILE = BASE_DIR / "test_normal_split.csv"
TEST_ATTACK_FILE = BASE_DIR / "test_attack_split.csv"
ATTACK_FILE = BASE_DIR / "attack_behavior.csv"
TEST_FILE = BASE_DIR / "test_behavior.csv"
TEST_ALIAS_FILE = BASE_DIR / "test_dataset.csv"
SPLIT_CONFIG_FILE = BASE_DIR / "cic_split_config.json"
SUMMARY_FILE = BASE_DIR / "cic_split_summary.txt"

TARGET = "target"
GROUP = "source_file"
RANDOM_STATE = 42
TARGET_PROPORTIONS = np.array([0.60, 0.20, 0.20], dtype=float)


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")


def normalize(df):
    out = df.copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    if GROUP in out.columns:
        out[GROUP] = out[GROUP].astype(str)
    return out


def group_partition(df):
    if GROUP not in df.columns:
        raise ValueError("behavior_dataset.csv ต้องมี source_file เพื่อทำ group-level split")

    group_rows = df.groupby(GROUP).size().to_dict()
    group_attack = df.groupby(GROUP)[TARGET].sum().to_dict()
    groups = list(group_rows)
    n_total = len(df)

    # CIC-IDS2017 behavior_dataset normally has a small number of PCAP groups.
    # Exhaustive search gives a deterministic and transparent split for <= 12 groups.
    if len(groups) > 12:
        raise ValueError(
            f"พบ source_file {len(groups)} กลุ่ม ซึ่งมากเกินกว่าการ split แบบ deterministic นี้ "
            "ให้ลดระดับกลุ่มหรือปรับ Stage 07 ก่อนใช้งาน"
        )

    best = None
    for train_bits in itertools.product([0, 1], repeat=len(groups)):
        train_groups = [groups[i] for i, b in enumerate(train_bits) if b]
        if not train_groups:
            continue
        # Train must be legitimate only.
        if any(group_attack[g] != 0 for g in train_groups):
            continue

        remaining = [g for g in groups if g not in train_groups]
        if len(remaining) < 2:
            continue

        for r in range(1, len(remaining)):
            for val_groups_tuple in itertools.combinations(remaining, r):
                val_groups = list(val_groups_tuple)
                test_groups = [g for g in remaining if g not in val_groups]
                if not test_groups:
                    continue

                train_df = df[df[GROUP].isin(train_groups)]
                val_df = df[df[GROUP].isin(val_groups)]
                test_df = df[df[GROUP].isin(test_groups)]

                # Validation must support the Hybrid IF + RF development split:
                # at least two source_file groups, and every validation group must
                # contain both BENIGN and ATTACK so either half can remain labeled.
                if len(val_groups) < 2 or val_df[TARGET].nunique() < 2 or test_df[TARGET].nunique() < 2:
                    continue
                valid_group_classes = df[df[GROUP].isin(val_groups)].groupby(GROUP)[TARGET].nunique()
                if len(valid_group_classes) != len(val_groups) or (valid_group_classes < 2).any():
                    continue

                proportions = np.array([
                    len(train_df), len(val_df), len(test_df)
                ], dtype=float) / n_total
                error = float(np.sum((proportions - TARGET_PROPORTIONS) ** 2))

                candidate = (error, proportions, train_groups, val_groups, test_groups)
                if best is None or error < best[0]:
                    best = candidate

    if best is None:
        raise RuntimeError(
            "ไม่พบการแบ่ง source_file ที่ทำให้ Train เป็น BENIGN-only "
            "และ Validation/Final Test มีทั้ง BENIGN+ATTACK"
        )

    _, proportions, train_groups, val_groups, test_groups = best
    train = df[df[GROUP].isin(train_groups)].copy()
    validation = df[df[GROUP].isin(val_groups)].copy()
    test = df[df[GROUP].isin(test_groups)].copy()

    # Stable row order inside each split.
    sort_cols = [c for c in [GROUP, "window_id"] if c in df.columns]
    if sort_cols:
        train = train.sort_values(sort_cols, kind="stable")
        validation = validation.sort_values(sort_cols, kind="stable")
        test = test.sort_values(sort_cols, kind="stable")

    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), train_groups, val_groups, test_groups, proportions


def main():
    require(INPUT_FILE)
    df = normalize(pd.read_csv(INPUT_FILE))
    if TARGET not in df.columns:
        raise ValueError("behavior_dataset.csv ต้องมี column target")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("target ต้องมีเฉพาะ 0=BENIGN และ 1=ATTACK")

    train, validation, test, train_groups, val_groups, test_groups, props = group_partition(df)

    if train[TARGET].sum() != 0:
        raise RuntimeError("Train leakage: พบ ATTACK ใน Train")
    if set(train[GROUP]) & set(validation[GROUP]):
        raise RuntimeError("Train / Validation source_file overlap")
    if set(train[GROUP]) & set(test[GROUP]):
        raise RuntimeError("Train / Final Test source_file overlap")
    if set(validation[GROUP]) & set(test[GROUP]):
        raise RuntimeError("Validation / Final Test source_file overlap")

    # Keep an attack pool file for backward compatibility, but the pool is NOT
    # used by later stages to build Final Test. Final Test comes only from TEST_FILE.
    attack_pool = df[df[TARGET] == 1].copy().reset_index(drop=True)
    test_normal = test[test[TARGET] == 0].copy().reset_index(drop=True)
    test_attack = test[test[TARGET] == 1].copy().reset_index(drop=True)

    train.to_csv(TRAIN_FILE, index=False)
    validation.to_csv(VALIDATION_FILE, index=False)
    test_normal.to_csv(TEST_NORMAL_FILE, index=False)
    test_attack.to_csv(TEST_ATTACK_FILE, index=False)
    attack_pool.to_csv(ATTACK_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    test.to_csv(TEST_ALIAS_FILE, index=False)

    config = {
        "group_column": GROUP,
        "random_state": RANDOM_STATE,
        "target_proportions": TARGET_PROPORTIONS.tolist(),
        "actual_proportions": props.tolist(),
        "train_groups": train_groups,
        "validation_groups": val_groups,
        "test_groups": test_groups,
        "validation_is_tuning_only": True,
        "final_test_is_untouched_until_stage_15_plus": True,
    }
    SPLIT_CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("CIC-IDS2017 CLEAN TRAIN / VALIDATION / FINAL TEST SPLIT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total rows = {len(df):,}\n")
        f.write(f"Train rows = {len(train):,}\n")
        f.write(f"Validation rows = {len(validation):,}\n")
        f.write(f"Final Test rows = {len(test):,}\n\n")
        f.write(f"Train groups = {train_groups}\n")
        f.write(f"Validation groups = {val_groups}\n")
        f.write(f"Final Test groups = {test_groups}\n\n")
        f.write(f"Train target = {train[TARGET].value_counts().to_dict()}\n")
        f.write(f"Validation target = {validation[TARGET].value_counts().to_dict()}\n")
        f.write(f"Final Test target = {test[TARGET].value_counts().to_dict()}\n")
        f.write("\nNo source_file appears in more than one split.\n")
        f.write("Attack pool is retained only for backward compatibility and is not sampled into Final Test.\n")

    print("=" * 90)
    print("CIC-IDS2017 STAGE 07 — CLEAN GROUP-BASED SPLIT")
    print("=" * 90)
    print(f"Total          : {len(df):,}")
    print(f"Train          : {len(train):,}  (BENIGN only)")
    print(f"Validation     : {len(validation):,}  {validation[TARGET].value_counts().sort_index().to_dict()}")
    print(f"Final Test     : {len(test):,}  {test[TARGET].value_counts().sort_index().to_dict()}")
    print()
    print(f"Train groups   : {train_groups}")
    print(f"Validation     : {val_groups}")
    print(f"Final Test     : {test_groups}")
    print()
    print("Group overlap  : PASS")
    print("Train ATTACK   : PASS (0)")
    print("Attack pool is NOT used to construct Final Test.")
    print("=" * 90)


if __name__ == "__main__":
    main()
