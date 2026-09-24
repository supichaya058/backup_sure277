from pathlib import Path
from collections import Counter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATTERN = "*-WorkingHours*.pcap_ISCX.csv"
CHUNK_SIZE = 50_000

RAW_FILES = sorted(BASE_DIR.glob(RAW_PATTERN))

print("=" * 80)
print("CIC-IDS2017 RAW DATA INSPECTION")
print("=" * 80)
print(f"Files: {len(RAW_FILES)}")
print()

if not RAW_FILES:
    raise FileNotFoundError("ไม่พบไฟล์ CIC-IDS2017 raw CSV")

reference_columns = None

for file_path in RAW_FILES:
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    total_rows = 0
    labels = Counter()
    first_chunk = True

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(f"{file_path.name} ไม่มี column Label")

        if first_chunk:
            reference_columns = list(chunk.columns)
            print(f"Columns: {len(reference_columns)}")
            print("Column names:")
            for i, col in enumerate(reference_columns, 1):
                print(f"  {i:02d}. {col}")
            print()
            first_chunk = False

        label_values = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )
        labels.update(label_values)
        total_rows += len(chunk)

    print(f"Total Rows: {total_rows:,}")
    print("Labels:")
    for label, count in labels.most_common():
        pct = 100 * count / total_rows if total_rows else 0
        print(f"  {label:<35} {count:>12,} ({pct:6.2f}%)")
    print()

print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
print("หมายเหตุ: Stage 04 จะวิเคราะห์ raw-flow features เชิงข้อมูล")
print("ส่วน feature selection สำหรับโมเดล Behavior จะทำจาก TRAIN เท่านั้นใน Stage 08")
