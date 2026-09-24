from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATTERN = "*-WorkingHours*.pcap_ISCX.csv"
CHUNK_SIZE = 25_000
ROWS_PER_FILE = 5_000

SUMMARY_FILE = BASE_DIR / "raw_flow_feature_summary.csv"
CORRELATION_FILE = BASE_DIR / "raw_flow_correlation_matrix.csv"

RAW_FILES = sorted(BASE_DIR.glob(RAW_PATTERN))

print("=" * 80)
print("CIC-IDS2017 RAW FLOW FEATURE ANALYSIS")
print("=" * 80)
print(f"Files: {len(RAW_FILES)}")
print()
print("Stage นี้เป็น diagnostic ของ raw-flow data เท่านั้น")
print("จะไม่สร้าง selected_features.csv และไม่ป้อน feature list ให้ Behavior Model")
print("Behavior-level feature selection จะทำจาก TRAIN เท่านั้นใน Stage 08")
print()

if not RAW_FILES:
    raise FileNotFoundError("ไม่พบไฟล์ CIC-IDS2017 raw CSV")

samples = []

for file_path in RAW_FILES:
    file_sample = []

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        numeric = chunk.select_dtypes(include=[np.number]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan)

        if len(numeric):
            take = min(len(numeric), ROWS_PER_FILE - len(file_sample))
            if take > 0:
                file_sample.append(numeric.head(take))

        if sum(len(x) for x in file_sample) >= ROWS_PER_FILE:
            break

    if file_sample:
        sampled = pd.concat(file_sample, ignore_index=True)
        sampled = sampled.head(ROWS_PER_FILE)
        samples.append(sampled)

if not samples:
    raise RuntimeError("สร้าง raw-flow sample ไม่ได้")

df = pd.concat(samples, ignore_index=True)

summary = pd.DataFrame({
    "feature": df.columns,
    "non_null": [int(df[c].notna().sum()) for c in df.columns],
    "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    "mean": [float(df[c].mean()) for c in df.columns],
    "std": [float(df[c].std(ddof=0)) for c in df.columns],
    "min": [float(df[c].min()) for c in df.columns],
    "max": [float(df[c].max()) for c in df.columns],
})

summary.to_csv(SUMMARY_FILE, index=False)

corr = df.corr(method="pearson")
corr.to_csv(CORRELATION_FILE)

print(f"Sampled raw-flow rows : {len(df):,}")
print(f"Numeric features      : {len(df.columns)}")
print()
print(f"Saved: {SUMMARY_FILE.name}")
print(f"Saved: {CORRELATION_FILE.name}")
print()
print("Stage 04 COMPLETE")
