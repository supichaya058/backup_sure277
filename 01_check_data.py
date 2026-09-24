from pathlib import Path
from collections import Counter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATTERN = "*-WorkingHours*.pcap_ISCX.csv"
CHUNK_SIZE = 50_000

RAW_FILES = sorted(BASE_DIR.glob(RAW_PATTERN))

print("=" * 80)
print("CIC-IDS2017 RAW DATA CHECK")
print("=" * 80)
print(f"Raw input pattern : {RAW_PATTERN}")
print(f"Raw files found   : {len(RAW_FILES)}")
print()

if not RAW_FILES:
    raise FileNotFoundError("ไม่พบไฟล์ CIC-IDS2017 raw CSV")

total_rows = 0

for file_path in RAW_FILES:
    rows = 0
    labels = Counter()
    columns = None

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        if columns is None:
            columns = list(chunk.columns)

        if "Label" not in chunk.columns:
            raise ValueError(f"{file_path.name} ไม่มี column Label")

        label_values = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )
        labels.update(label_values)
        rows += len(chunk)

    total_rows += rows

    print(f"{file_path.name}")
    print(f"  Rows    : {rows:,}")
    print(f"  Columns : {len(columns)}")
    print(f"  Size MB : {file_path.stat().st_size / (1024**2):.2f}")
    print("  Labels  :")
    for label, count in labels.most_common():
        print(f"    {label:<35} {count:>12,}")
    print()

print("=" * 80)
print("RAW DATA TOTAL")
print("=" * 80)
print(f"Files : {len(RAW_FILES)}")
print(f"Rows  : {total_rows:,}")
print("=" * 80)
