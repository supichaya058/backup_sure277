from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# CIC-IDS2017 -> 13 BEHAVIORAL FEATURES
# ============================================================
#
# Framework จากงานเดิม:
#
# Volume
#   1. Request Rate
#   2. Total Requests
#   3. Throughput
#   4. Flow Count
#
# Error
#   5. Error Rate
#   6. 4xx Count
#   7. 5xx Count
#
# Latency
#   8. Avg Latency
#   9. Max Latency
#  10. Latency Std Dev
#
# Source
#  11. Unique IP Count
#  12. New IP Ratio
#
# Connection
#  13. Active Sessions
#
# ------------------------------------------------------------
# หลักการสำคัญ
# ------------------------------------------------------------
# CIC CSV ชุดปัจจุบันที่เราใช้อยู่ไม่มี Source IP / Timestamp /
# HTTP Status Code ใน 79 columns ที่ตรวจพบ
#
# ดังนั้น:
#   - Feature ที่คำนวณได้จาก CIC จะถูกคำนวณจริง
#   - Feature ที่ข้อมูลไม่รองรับจะเป็น NaN ในไฟล์ 05
#   - จะไม่มีการเอา Label ไปสร้าง Error Rate
#   - จะไม่มีการเดา IP หรือ HTTP status ขึ้นมาเอง
#
# Stage 06 จะเลือกเฉพาะ Feature ที่ข้อมูลรองรับจริงไปสร้าง
# behavior_dataset.csv เพื่อให้ Stage 07-15 เดินต่อได้
#
# Window:
#   50 flows = 1 behavior window
#   Mixed BENIGN + ATTACK window ถูกตัดออก
#   Partial window (< 50 flows) ถูกตัดออก
#
# Output:
#   behavior_features.csv
#   behavior_13_feature_definition.csv
#   behavior_13_feature_availability.csv
#   behavior_summary.csv
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = sorted(
    BASE_DIR.glob("*-WorkingHours*.pcap_ISCX.csv")
)

OUTPUT_FILE = BASE_DIR / "behavior_features.csv"
DEFINITION_FILE = BASE_DIR / "behavior_13_feature_definition.csv"
AVAILABILITY_FILE = BASE_DIR / "behavior_13_feature_availability.csv"
SUMMARY_FILE = BASE_DIR / "behavior_summary.csv"

WINDOW_SIZE = 50
CHUNK_SIZE = 50_000

LABEL_COL = "Label"

SOURCE_IP_CANDIDATES = [
    "Source IP",
    "Src IP",
    "source_ip",
    "src_ip",
]

TIMESTAMP_CANDIDATES = [
    "Timestamp",
    "timestamp",
    "Time",
    "time",
]

STATUS_CANDIDATES = [
    "Status Code",
    "HTTP Status",
    "HTTP Status Code",
    "status_code",
    "status",
]

LATENCY_CANDIDATES = [
    "Latency",
    "Response Time",
    "Response_Time",
    "response_time",
    "Request Latency",
    "request_latency",
]


BEHAVIOR_13 = [
    ("Volume", "Request Rate", "request_rate"),
    ("Volume", "Total Requests", "total_requests"),
    ("Volume", "Throughput", "throughput"),
    ("Volume", "Flow Count", "flow_count_feature"),

    ("Error", "Error Rate", "error_rate"),
    ("Error", "4xx Count", "4xx_count"),
    ("Error", "5xx Count", "5xx_count"),

    ("Latency", "Avg Latency", "avg_latency"),
    ("Latency", "Max Latency", "max_latency"),
    ("Latency", "Latency Std Dev", "latency_std_dev"),

    ("Source", "Unique IP Count", "unique_ip_count"),
    ("Source", "New IP Ratio", "new_ip_ratio"),

    ("Connection", "Active Sessions", "active_sessions"),
]


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def safe_mean(series):
    value = series.mean(skipna=True)
    return float(value) if pd.notna(value) else np.nan


def safe_std(series):
    value = series.std(
        ddof=0,
        skipna=True,
    )
    return float(value) if pd.notna(value) else np.nan


def safe_max(series):
    value = series.max(skipna=True)
    return float(value) if pd.notna(value) else np.nan


def safe_min(series):
    value = series.min(skipna=True)
    return float(value) if pd.notna(value) else np.nan


def total_bytes(group):
    fwd = numeric_series(
        group,
        "Total Length of Fwd Packets",
    ).fillna(0.0)

    bwd = numeric_series(
        group,
        "Total Length of Bwd Packets",
    ).fillna(0.0)

    return float((fwd + bwd).sum())


def total_duration_seconds(group):
    duration = numeric_series(
        group,
        "Flow Duration",
    ).fillna(0.0).clip(lower=0.0)

    return float(duration.sum() / 1_000_000.0)


def timestamp_window_seconds(group, timestamp_col):
    if timestamp_col is None:
        return np.nan

    ts = pd.to_datetime(
        group[timestamp_col],
        errors="coerce",
    ).dropna()

    if len(ts) < 2:
        return np.nan

    seconds = (
        ts.max() - ts.min()
    ).total_seconds()

    if seconds <= 0:
        return np.nan

    return float(seconds)


def window_duration_seconds(group, timestamp_col):
    # Preferred: real timestamp span.
    timestamp_seconds = timestamp_window_seconds(
        group,
        timestamp_col,
    )

    if pd.notna(timestamp_seconds):
        return timestamp_seconds, "timestamp_span"

    # Current CIC CSV fallback:
    # aggregate flow duration, because no Timestamp exists.
    duration_seconds = total_duration_seconds(group)

    if duration_seconds > 0:
        return duration_seconds, "sum_flow_duration"

    return np.nan, "unavailable"


def calculate_active_sessions(
    group,
    source_ip_col,
    timestamp_col,
):
    if (
        source_ip_col is None
        or timestamp_col is None
        or "Flow Duration" not in group.columns
    ):
        return np.nan

    temp = group[
        [
            source_ip_col,
            timestamp_col,
            "Flow Duration",
        ]
    ].copy()

    temp[timestamp_col] = pd.to_datetime(
        temp[timestamp_col],
        errors="coerce",
    )

    temp["Flow Duration"] = pd.to_numeric(
        temp["Flow Duration"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    temp = temp.dropna(
        subset=[
            source_ip_col,
            timestamp_col,
        ]
    )

    if temp.empty:
        return np.nan

    events = []

    for _, row in temp.iterrows():
        start = row[timestamp_col]
        end = (
            start
            + pd.to_timedelta(
                float(row["Flow Duration"]) / 1_000_000.0,
                unit="s",
            )
        )

        events.append(
            (start, +1)
        )
        events.append(
            (end, -1)
        )

    # End events first when two events share a timestamp.
    events.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    active = 0
    maximum = 0

    for _, delta in events:
        active += delta
        maximum = max(
            maximum,
            active,
        )

    return int(maximum)


def calculate_new_ip_ratio(
    group,
    source_ip_col,
    seen_ips,
):
    if source_ip_col is None:
        return np.nan

    values = (
        group[source_ip_col]
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
        & values.ne("nan")
        & values.ne("None")
    ]

    unique_ips = list(
        dict.fromkeys(values.tolist())
    )

    if not unique_ips:
        return np.nan

    new_ips = [
        ip
        for ip in unique_ips
        if ip not in seen_ips
    ]

    seen_ips.update(unique_ips)

    return float(
        len(new_ips) / len(unique_ips)
    )


def build_behavior_row(
    group,
    source_file,
    window_id,
    seen_ips,
):
    flow_count = len(group)

    source_ip_col = first_existing_column(
        group,
        SOURCE_IP_CANDIDATES,
    )

    timestamp_col = first_existing_column(
        group,
        TIMESTAMP_CANDIDATES,
    )

    status_col = first_existing_column(
        group,
        STATUS_CANDIDATES,
    )

    latency_col = first_existing_column(
        group,
        LATENCY_CANDIDATES,
    )

    # --------------------------------------------------------
    # Labels / purity
    # --------------------------------------------------------

    labels = (
        group[LABEL_COL]
        .astype(str)
        .str.strip()
    )

    attack_mask = (
        labels.str.upper() != "BENIGN"
    )

    attack_flow_count = int(
        attack_mask.sum()
    )

    benign_flow_count = int(
        (~attack_mask).sum()
    )

    attack_ratio = (
        attack_flow_count / flow_count
        if flow_count
        else 0.0
    )

    if attack_flow_count == 0:
        target = 0
        behavior_label = "BENIGN"
        attack_type = "BENIGN"
        window_purity = "PURE_BENIGN"

    elif attack_flow_count == flow_count:
        target = 1

        unique_labels = sorted(
            labels.unique().tolist()
        )

        if len(unique_labels) == 1:
            behavior_label = unique_labels[0]
            attack_type = unique_labels[0]
        else:
            behavior_label = "MULTIPLE_ATTACKS"
            attack_type = "MULTIPLE_ATTACKS"

        window_purity = "PURE_ATTACK"

    else:
        raise ValueError(
            "Mixed window ถูกส่งเข้ามา "
            "ทั้งที่ควรถูก discard ก่อน"
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    window_seconds, window_method = (
        window_duration_seconds(
            group,
            timestamp_col,
        )
    )

    if pd.notna(window_seconds) and window_seconds > 0:
        request_rate = (
            float(flow_count / window_seconds)
        )

        throughput = (
            total_bytes(group) / window_seconds
        )
    else:
        request_rate = np.nan
        throughput = np.nan

    total_requests = flow_count
    flow_count_feature = flow_count

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------
    #
    # Only calculate when actual HTTP status information exists.
    # Never use Label to create Error Rate.
    # --------------------------------------------------------

    if status_col is not None:
        status = pd.to_numeric(
            group[status_col],
            errors="coerce",
        )

        status = status.dropna()

        count_4xx = int(
            (
                (status >= 400)
                & (status <= 499)
            ).sum()
        )

        count_5xx = int(
            (
                (status >= 500)
                & (status <= 599)
            ).sum()
        )

        status_count = len(status)

        error_rate = (
            (count_4xx + count_5xx) / status_count
            if status_count > 0
            else np.nan
        )
    else:
        count_4xx = np.nan
        count_5xx = np.nan
        error_rate = np.nan

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    if latency_col is not None:
        latency = numeric_series(
            group,
            latency_col,
        ).dropna()

        avg_latency = safe_mean(latency)
        max_latency = safe_max(latency)
        latency_std_dev = safe_std(latency)
        latency_method = (
            f"actual:{latency_col}"
        )
    else:
        # Current CIC CSV has Flow Duration.
        latency = numeric_series(
            group,
            "Flow Duration",
        ).dropna()

        avg_latency = safe_mean(latency)
        max_latency = safe_max(latency)
        latency_std_dev = safe_std(latency)
        latency_method = "proxy:Flow Duration"

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if source_ip_col is not None:
        source_values = (
            group[source_ip_col]
            .astype(str)
            .str.strip()
        )

        source_values = source_values[
            source_values.ne("")
            & source_values.ne("nan")
            & source_values.ne("None")
        ]

        unique_ip_count = int(
            source_values.nunique()
        )

        new_ip_ratio = calculate_new_ip_ratio(
            group,
            source_ip_col,
            seen_ips,
        )
    else:
        unique_ip_count = np.nan
        new_ip_ratio = np.nan

    # --------------------------------------------------------
    # Connection
    # --------------------------------------------------------

    active_sessions = calculate_active_sessions(
        group,
        source_ip_col,
        timestamp_col,
    )

    return {
        # Metadata / target
        "source_file": source_file,
        "window_id": window_id,
        "flow_count": flow_count,
        "benign_flow_count": benign_flow_count,
        "attack_flow_count": attack_flow_count,
        "attack_ratio": attack_ratio,
        "window_purity": window_purity,
        "target": target,
        "behavior_label": behavior_label,
        "attack_type": attack_type,

        # 13 Behavioral Features
        "request_rate": request_rate,
        "total_requests": total_requests,
        "throughput": throughput,
        "flow_count_feature": flow_count_feature,

        "error_rate": error_rate,
        "4xx_count": count_4xx,
        "5xx_count": count_5xx,

        "avg_latency": avg_latency,
        "max_latency": max_latency,
        "latency_std_dev": latency_std_dev,

        "unique_ip_count": unique_ip_count,
        "new_ip_ratio": new_ip_ratio,

        "active_sessions": active_sessions,

        # Methodology / audit fields
        "window_duration_method": window_method,
        "latency_method": latency_method,
        "source_ip_column_used": (
            source_ip_col or ""
        ),
        "timestamp_column_used": (
            timestamp_col or ""
        ),
        "status_column_used": (
            status_col or ""
        ),
        "latency_column_used": (
            latency_col or ""
        ),
    }


def process_file(file_path):
    print()
    print("=" * 80)
    print(f"PROCESSING: {file_path.name}")
    print("=" * 80)

    rows = []
    carry = pd.DataFrame()

    window_id = 0
    flow_rows = 0
    mixed_windows_discarded = 0
    partial_windows_discarded = 0

    # Needed for "New IP Ratio" when Source IP exists.
    seen_ips = set()

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk.columns = (
            chunk.columns
            .astype(str)
            .str.strip()
        )

        if LABEL_COL not in chunk.columns:
            raise ValueError(
                f"{file_path.name} ไม่มี column '{LABEL_COL}'"
            )

        chunk[LABEL_COL] = (
            chunk[LABEL_COL]
            .astype(str)
            .str.strip()
        )

        chunk = chunk[
            chunk[LABEL_COL].ne("")
            & chunk[LABEL_COL].ne("nan")
        ].copy()

        for col in [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
        ]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(
                    chunk[col],
                    errors="coerce",
                )

        flow_rows += len(chunk)

        if not carry.empty:
            chunk = pd.concat(
                [carry, chunk],
                ignore_index=True,
            )
            carry = pd.DataFrame()

        full_windows = (
            len(chunk) // WINDOW_SIZE
        )

        usable_rows = (
            full_windows * WINDOW_SIZE
        )

        if usable_rows == 0:
            carry = chunk.copy()
            continue

        usable = chunk.iloc[
            :usable_rows
        ].copy()

        carry = chunk.iloc[
            usable_rows:
        ].copy()

        for start in range(
            0,
            len(usable),
            WINDOW_SIZE,
        ):
            end = start + WINDOW_SIZE

            group = usable.iloc[
                start:end
            ].copy()

            labels = (
                group[LABEL_COL]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            attack_mask = (
                labels != "BENIGN"
            )

            if (
                attack_mask.any()
                and (~attack_mask).any()
            ):
                mixed_windows_discarded += 1
                continue

            window_id += 1

            rows.append(
                build_behavior_row(
                    group=group,
                    source_file=file_path.name,
                    window_id=window_id,
                    seen_ips=seen_ips,
                )
            )

    if not carry.empty:
        partial_windows_discarded += 1

    result = pd.DataFrame(rows)

    print(
        f"Flow rows               : {flow_rows:,}"
    )
    print(
        f"Behavior windows        : {len(result):,}"
    )
    print(
        f"Mixed windows discarded : "
        f"{mixed_windows_discarded:,}"
    )
    print(
        f"Partial window discarded: "
        f"{partial_windows_discarded:,}"
    )

    return result


def main():
    print("=" * 80)
    print("CIC-IDS2017 - CREATE 13 BEHAVIORAL FEATURES")
    print("=" * 80)

    if not INPUT_FILES:
        raise FileNotFoundError(
            "ไม่พบไฟล์ CIC-IDS2017 "
            "รูปแบบ *-WorkingHours*.pcap_ISCX.csv"
        )

    print(
        f"Input files : {len(INPUT_FILES)}"
    )
    print(
        f"Window size : {WINDOW_SIZE} flows"
    )
    print(
        "Window type : flow-based"
    )
    print(
        "Mixed rule  : discard mixed BENIGN + ATTACK"
    )
    print(
        "Partial rule: discard incomplete windows"
    )
    print()

    all_behavior = []

    for file_path in INPUT_FILES:
        result = process_file(
            file_path
        )

        if not result.empty:
            all_behavior.append(result)

    if not all_behavior:
        raise RuntimeError(
            "ไม่สามารถสร้าง Behavior Features ได้"
        )

    behavior_df = pd.concat(
        all_behavior,
        ignore_index=True,
    )

    behavior_df = behavior_df.sort_values(
        [
            "source_file",
            "window_id",
        ]
    ).reset_index(
        drop=True
    )

    behavior_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Definition table
    # --------------------------------------------------------

    definitions = [
        {
            "group": "Volume",
            "feature": "Request Rate",
            "column": "request_rate",
            "definition": (
                "จำนวน flow ต่อหน่วยเวลาของ behavior window"
            ),
            "cic_method": (
                "flow_count / window_duration_seconds; "
                "ใช้ Timestamp span หากมี มิฉะนั้น sum Flow Duration"
            ),
        },
        {
            "group": "Volume",
            "feature": "Total Requests",
            "column": "total_requests",
            "definition": (
                "จำนวน flow ใน behavior window "
                "ใช้เป็น request-level proxy"
            ),
            "cic_method": "flow_count",
        },
        {
            "group": "Volume",
            "feature": "Throughput",
            "column": "throughput",
            "definition": "Total bytes ต่อหน่วยเวลา",
            "cic_method": (
                "total bytes / window duration"
            ),
        },
        {
            "group": "Volume",
            "feature": "Flow Count",
            "column": "flow_count_feature",
            "definition": "จำนวน flow ใน behavior window",
            "cic_method": "count of rows in window",
        },
        {
            "group": "Error",
            "feature": "Error Rate",
            "column": "error_rate",
            "definition": (
                "(4xx + 5xx) / status observations"
            ),
            "cic_method": (
                "คำนวณเมื่อมี HTTP Status column เท่านั้น"
            ),
        },
        {
            "group": "Error",
            "feature": "4xx Count",
            "column": "4xx_count",
            "definition": "จำนวน HTTP status 400-499",
            "cic_method": (
                "คำนวณเมื่อมี HTTP Status column เท่านั้น"
            ),
        },
        {
            "group": "Error",
            "feature": "5xx Count",
            "column": "5xx_count",
            "definition": "จำนวน HTTP status 500-599",
            "cic_method": (
                "คำนวณเมื่อมี HTTP Status column เท่านั้น"
            ),
        },
        {
            "group": "Latency",
            "feature": "Avg Latency",
            "column": "avg_latency",
            "definition": "ค่าเฉลี่ย latency ของ flow ใน window",
            "cic_method": (
                "ใช้ Latency/Response Time หากมี; "
                "CIC ปัจจุบันใช้ Flow Duration เป็น proxy"
            ),
        },
        {
            "group": "Latency",
            "feature": "Max Latency",
            "column": "max_latency",
            "definition": "latency สูงสุดใน window",
            "cic_method": (
                "ใช้ Latency/Response Time หากมี; "
                "CIC ปัจจุบันใช้ Flow Duration เป็น proxy"
            ),
        },
        {
            "group": "Latency",
            "feature": "Latency Std Dev",
            "column": "latency_std_dev",
            "definition": "ส่วนเบี่ยงเบนมาตรฐานของ latency",
            "cic_method": (
                "ใช้ Latency/Response Time หากมี; "
                "CIC ปัจจุบันใช้ Flow Duration เป็น proxy"
            ),
        },
        {
            "group": "Source",
            "feature": "Unique IP Count",
            "column": "unique_ip_count",
            "definition": "จำนวน Source IP ที่ไม่ซ้ำใน window",
            "cic_method": (
                "คำนวณเมื่อมี Source IP column เท่านั้น"
            ),
        },
        {
            "group": "Source",
            "feature": "New IP Ratio",
            "column": "new_ip_ratio",
            "definition": (
                "จำนวน Source IP ใหม่ใน window / "
                "จำนวน unique Source IP ใน window"
            ),
            "cic_method": (
                "คำนวณเมื่อมี Source IP column เท่านั้น"
            ),
        },
        {
            "group": "Connection",
            "feature": "Active Sessions",
            "column": "active_sessions",
            "definition": "จำนวน session ที่ active พร้อมกัน",
            "cic_method": (
                "ต้องมี Source IP + Timestamp + Flow Duration"
            ),
        },
    ]

    definition_df = pd.DataFrame(
        definitions
    )

    definition_df.to_csv(
        DEFINITION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Availability table
    # --------------------------------------------------------

    availability_rows = []

    for group, feature, column in BEHAVIOR_13:
        if column not in behavior_df.columns:
            status = "NOT_CREATED"
            non_null = 0
        else:
            non_null = int(
                behavior_df[column].notna().sum()
            )

            if non_null == len(behavior_df):
                status = "AVAILABLE"
            elif non_null == 0:
                status = "UNAVAILABLE"
            else:
                status = "PARTIAL"

        availability_rows.append(
            {
                "group": group,
                "feature": feature,
                "column": column,
                "status": status,
                "non_null_rows": non_null,
                "total_rows": len(behavior_df),
                "model_eligible": (
                    status == "AVAILABLE"
                ),
            }
        )

    availability_df = pd.DataFrame(
        availability_rows
    )

    availability_df.to_csv(
        AVAILABILITY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        behavior_df
        .groupby(
            [
                "target",
                "window_purity",
                "behavior_label",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="behavior_count"
        )
        .sort_values(
            [
                "target",
                "behavior_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("13 BEHAVIORAL FEATURE AVAILABILITY")
    print("=" * 80)
    print(
        availability_df[
            [
                "group",
                "feature",
                "status",
                "non_null_rows",
                "total_rows",
                "model_eligible",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Total behavior windows : "
        f"{len(behavior_df):,}"
    )

    print()
    print("Saved:")
    print(
        f"  {OUTPUT_FILE.name}"
    )
    print(
        f"  {DEFINITION_FILE.name}"
    )
    print(
        f"  {AVAILABILITY_FILE.name}"
    )
    print(
        f"  {SUMMARY_FILE.name}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
