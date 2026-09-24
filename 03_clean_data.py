from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATTERN = "*-WorkingHours*.pcap_ISCX.csv"
CHUNK_SIZE = 50_000

RAW_FILES = sorted(BASE_DIR.glob(RAW_PATTERN))

print("=" * 80)
print("CIC-IDS2017 CLEANING CHECK")
print("=" * 80)
print(f"Files checked: {len(RAW_FILES)}")
print()

if not RAW_FILES:
    raise FileNotFoundError("ไม่พบไฟล์ CIC-IDS2017 raw CSV")

total_rows = 0
total_missing = 0
total_inf = 0

for file_path in RAW_FILES:
    file_rows = 0
    file_missing = 0
    file_inf = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()
        file_rows += len(chunk)

        file_missing += int(chunk.isna().sum().sum())

        numeric_cols = chunk.select_dtypes(
            include=[np.number]
        ).columns

        if len(numeric_cols):
            values = chunk[numeric_cols].to_numpy()
            file_inf += int(np.isinf(values).sum())

    total_rows += file_rows
    total_missing += file_missing
    total_inf += file_inf

    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)
    print(f"Rows       : {file_rows:,}")
    print(f"Missing    : {file_missing:,}")
    print(f"Infinity   : {file_inf:,}")

print("=" * 80)
print("TOTAL")
print("=" * 80)
print(f"Rows       : {total_rows:,}")
print(f"Missing    : {total_missing:,}")
print(f"Infinity   : {total_inf:,}")
print("=" * 80)
print("Cleaning policy used by Stage 05:")
print("- Numeric Inf -> NaN")
print("- Missing numeric values -> 0")
print("- Empty/missing labels -> discarded")
print("- No generated CSV is treated as a raw input")
print("=" * 80)
