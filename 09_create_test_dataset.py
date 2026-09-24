from pathlib import Path
import pandas as pd

# ============================================================
# STAGE 09: FINAL TEST DATASET
# ============================================================
# Stage 07 has already created a clean, untouched Final Test.
# This stage only validates/copies that split for backward-compatible
# downstream names.
# NO attack sampling is performed here.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "test_behavior.csv"
OUTPUT_FILE = BASE_DIR / "test_dataset.csv"
SUMMARY_FILE = BASE_DIR / "test_dataset_summary.txt"
SPLIT_CONFIG = BASE_DIR / "cic_split_config.json"


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")


for p in [TEST_FILE, SPLIT_CONFIG]:
    require(p)

test = pd.read_csv(TEST_FILE)
if "target" not in test.columns:
    raise ValueError("test_behavior.csv ต้องมี column target")

test["target"] = pd.to_numeric(test["target"], errors="coerce").fillna(0).astype(int)
if set(test["target"].unique()) - {0, 1}:
    raise ValueError("Final Test target ต้องเป็น 0/1")

# Preserve stable evaluation order.
test = test.reset_index(drop=True)
test["evaluation_order"] = range(len(test))
test.to_csv(TEST_FILE, index=False)
test.to_csv(OUTPUT_FILE, index=False)

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("CIC-IDS2017 FINAL TEST DATASET\n")
    f.write("=" * 70 + "\n")
    f.write(f"Final Test rows = {len(test):,}\n")
    f.write(f"BENIGN = {(test['target']==0).sum():,}\n")
    f.write(f"ATTACK = {(test['target']==1).sum():,}\n")
    f.write("Construction = Stage 07 group-based Final Test; no attack sampling in Stage 09.\n")
    f.write("Final Test is reserved for final evaluation only.\n")

print("=" * 80)
print("CIC-IDS2017 STAGE 09 — FINAL TEST READY")
print("=" * 80)
print(f"Final Test rows : {len(test):,}")
print(f"BENIGN          : {(test['target']==0).sum():,}")
print(f"ATTACK          : {(test['target']==1).sum():,}")
print("Attack sampling : NO")
print("Final Test used : evaluation only")
print(f"Saved            : {OUTPUT_FILE}")
print("=" * 80)
