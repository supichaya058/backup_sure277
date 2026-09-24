from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CIC-IDS2017 -> BEHAVIOR DATASET
# ============================================================
#
# Input:
#   behavior_features.csv
#   behavior_13_feature_definition.csv
#   behavior_13_feature_availability.csv
#
# Output:
#   behavior_dataset.csv
#   behavior_feature_summary.csv
#   behavior_descriptive_features.csv
#   behavior_correlation_matrix.csv
#   behavior_model_features.csv
#
# หลักการ:
#   - 05 สร้างกรอบ 13 Behavioral Features
#   - Feature ที่ CIC รองรับจริงจะเข้า behavior_dataset.csv
#   - Feature ที่ CIC ไม่สามารถสังเกตได้จะไม่ถูกยัดค่า 0 หลอก ๆ
#   - flow_count เป็น metadata/constant เพราะทุก window มี 50 flows
#   - Feature Selection ของ Model ยังไม่ทำที่นี่
#   - Stage 08 เป็นผู้ทำ TRAIN-only feature selection
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "behavior_features.csv"
)

AVAILABILITY_FILE = (
    BASE_DIR / "behavior_13_feature_availability.csv"
)

DEFINITION_FILE = (
    BASE_DIR / "behavior_13_feature_definition.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "behavior_dataset.csv"
)

DESCRIPTIVE_FEATURE_FILE = (
    BASE_DIR / "behavior_descriptive_features.csv"
)

CORRELATION_FILE = (
    BASE_DIR / "behavior_correlation_matrix.csv"
)

SUMMARY_FILE = (
    BASE_DIR / "behavior_feature_summary.csv"
)

MODEL_FEATURE_FILE = (
    BASE_DIR / "behavior_model_features.csv"
)

CORRELATION_THRESHOLD = 0.90

RANDOM_STATE = 42


META_COLUMNS = [
    "source_file",
    "window_id",
    "flow_count",
    "benign_flow_count",
    "attack_flow_count",
    "attack_ratio",
    "window_purity",
    "target",
    "behavior_label",
    "attack_type",
    "window_duration_method",
    "latency_method",
    "source_ip_column_used",
    "timestamp_column_used",
    "status_column_used",
    "latency_column_used",
]


# Mapping names used by Stage 05 -> clean model names.
# Keep flow_count as metadata, because it is constant by design.
FEATURE_COLUMNS_13 = [
    "request_rate",
    "total_requests",
    "throughput",
    "flow_count_feature",
    "error_rate",
    "4xx_count",
    "5xx_count",
    "avg_latency",
    "max_latency",
    "latency_std_dev",
    "unique_ip_count",
    "new_ip_ratio",
    "active_sessions",
]


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {path}\n"
            "กรุณารัน Stage 05 ก่อน"
        )


def main():
    print("=" * 80)
    print("CIC-IDS2017 BEHAVIOR DATASET")
    print("=" * 80)

    require_file(INPUT_FILE)
    require_file(AVAILABILITY_FILE)
    require_file(DEFINITION_FILE)

    df = pd.read_csv(
        INPUT_FILE
    )

    availability_df = pd.read_csv(
        AVAILABILITY_FILE
    )

    print(
        f"Input rows   : {len(df):,}"
    )
    print(
        f"Input columns: {len(df.columns)}"
    )
    print()

    required_columns = [
        "source_file",
        "window_id",
        "flow_count",
        "benign_flow_count",
        "attack_flow_count",
        "attack_ratio",
        "window_purity",
        "target",
        "behavior_label",
        "attack_type",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "ไม่พบ columns ที่จำเป็น:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Normalize labels / metadata
    # --------------------------------------------------------

    df["target"] = (
        pd.to_numeric(
            df["target"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["window_id"] = (
        pd.to_numeric(
            df["window_id"],
            errors="coerce",
        )
    )

    df["flow_count"] = (
        pd.to_numeric(
            df["flow_count"],
            errors="coerce",
        )
    )

    if (
        df["window_id"].isna().any()
        or df["flow_count"].isna().any()
    ):
        raise ValueError(
            "พบค่า window_id หรือ flow_count ที่ไม่ใช่ตัวเลข"
        )

    df["window_id"] = (
        df["window_id"]
        .astype(int)
    )

    df["flow_count"] = (
        df["flow_count"]
        .astype(int)
    )

    for col in [
        "source_file",
        "behavior_label",
        "attack_type",
        "window_purity",
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Determine features actually supported by CIC.
    # --------------------------------------------------------

    available = (
        availability_df[
            availability_df["status"].eq(
                "AVAILABLE"
            )
            & availability_df["model_eligible"].astype(bool)
        ]["column"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    available = [
        col
        for col in available
        if col in FEATURE_COLUMNS_13
    ]

    unavailable = [
        col
        for col in FEATURE_COLUMNS_13
        if col not in available
    ]

    print(
        "13-framework feature status:"
    )
    for col in FEATURE_COLUMNS_13:
        status_row = availability_df[
            availability_df["column"].eq(col)
        ]

        if status_row.empty:
            status = "NOT_FOUND"
        else:
            status = status_row.iloc[0][
                "status"
            ]

        print(
            f"  {col:<24} {status}"
        )

    print()
    print(
        "Features available for dataset/model:"
    )
    for col in available:
        print(
            f"  - {col}"
        )

    print()
    print(
        "Features unavailable in current CIC CSV:"
    )
    for col in unavailable:
        print(
            f"  - {col}"
        )

    # --------------------------------------------------------
    # Rename flow_count_feature -> flow_count
    # for the 13-feature framework.
    #
    # Keep an explicit metadata flow_count column too.
    # The 13-feature version is therefore represented by:
    #   flow_count_feature -> flow_count_model_candidate
    # but it is later excluded as a constant.
    # --------------------------------------------------------

    if "flow_count_feature" in df.columns:
        df["flow_count_model_candidate"] = (
            pd.to_numeric(
                df["flow_count_feature"],
                errors="coerce",
            )
        )

    # --------------------------------------------------------
    # Select output columns.
    # --------------------------------------------------------

    keep_columns = []

    for col in [
        "source_file",
        "window_id",
        "flow_count",
        "benign_flow_count",
        "attack_flow_count",
        "attack_ratio",
        "window_purity",
        "target",
        "behavior_label",
        "attack_type",
    ]:
        if col in df.columns:
            keep_columns.append(col)

    # Use only truly available Behavioral Features.
    #
    # Do not pass unsupported NaN columns to Stage 08.
    keep_columns.extend(
        [
            col
            for col in available
            if col in df.columns
        ]
    )

    # Preserve the original 13-feature flow-count concept
    # as a separate metadata candidate, but do not make it a
    # model feature.
    if (
        "flow_count_model_candidate" in df.columns
        and "flow_count_model_candidate" not in keep_columns
    ):
        keep_columns.append(
            "flow_count_model_candidate"
        )

    behavior_dataset = (
        df[keep_columns]
        .copy()
    )

    # --------------------------------------------------------
    # Numeric cleaning for available features only.
    # --------------------------------------------------------

    numeric_feature_columns = [
        col
        for col in available
        if col in behavior_dataset.columns
    ]

    for col in numeric_feature_columns:
        behavior_dataset[col] = pd.to_numeric(
            behavior_dataset[col],
            errors="coerce",
        )

    bad_numeric = (
        behavior_dataset[
            numeric_feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .isna()
        .sum()
    )

    bad_numeric = (
        bad_numeric[
            bad_numeric > 0
        ]
    )

    if not bad_numeric.empty:
        raise ValueError(
            "Behavior Features ที่ควรคำนวณได้พบ NaN/Inf:\n"
            + bad_numeric.to_string()
        )

    # --------------------------------------------------------
    # Save Behavior Dataset
    # --------------------------------------------------------

    behavior_dataset = (
        behavior_dataset
        .sort_values(
            [
                "source_file",
                "window_id",
            ]
        )
        .reset_index(drop=True)
    )

    behavior_dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Descriptive feature summary
    # --------------------------------------------------------

    descriptive_rows = []

    for col in numeric_feature_columns:
        series = pd.to_numeric(
            behavior_dataset[col],
            errors="coerce",
        )

        descriptive_rows.append(
            {
                "feature": col,
                "count": int(series.notna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "constant": bool(
                    series.nunique(dropna=True) <= 1
                ),
            }
        )

    descriptive_df = pd.DataFrame(
        descriptive_rows
    )

    descriptive_df.to_csv(
        DESCRIPTIVE_FEATURE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    if numeric_feature_columns:
        corr = (
            behavior_dataset[
                numeric_feature_columns
            ]
            .corr(
                method="pearson"
            )
        )
    else:
        corr = pd.DataFrame()

    corr.to_csv(
        CORRELATION_FILE
    )

    # --------------------------------------------------------
    # Model feature candidates
    #
    # Important:
    # Stage 08 still performs the real TRAIN-only selection.
    # This file is descriptive / diagnostic only.
    # --------------------------------------------------------

    model_candidates = descriptive_df[
        ~descriptive_df["constant"]
    ][
        [
            "feature",
            "count",
            "unique",
            "constant",
        ]
    ].copy()

    model_candidates.to_csv(
        MODEL_FEATURE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    summary_rows = []

    for _, row in availability_df.iterrows():
        col = row["column"]

        if (
            col in behavior_dataset.columns
            and col in numeric_feature_columns
        ):
            series = behavior_dataset[col]

            summary_rows.append(
                {
                    "group": row["group"],
                    "feature": row["feature"],
                    "column": col,
                    "status": row["status"],
                    "model_eligible": row["model_eligible"],
                    "count": int(series.notna().sum()),
                    "unique": int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                    "constant": bool(
                        series.nunique(
                            dropna=True
                        ) <= 1
                    ),
                }
            )
        else:
            summary_rows.append(
                {
                    "group": row["group"],
                    "feature": row["feature"],
                    "column": col,
                    "status": row["status"],
                    "model_eligible": False,
                    "count": 0,
                    "unique": 0,
                    "constant": False,
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BEHAVIOR DATASET RESULT")
    print("=" * 80)

    print(
        f"Behavior rows : "
        f"{len(behavior_dataset):,}"
    )

    print(
        f"Available features : "
        f"{len(numeric_feature_columns):,}"
    )

    print()
    print(
        "Available Behavioral Features:"
    )
    print(
        "\n".join(
            f"  {i}. {name}"
            for i, name in enumerate(
                numeric_feature_columns,
                1,
            )
        )
    )

    print()
    print(
        "หมายเหตุ:"
    )
    print(
        "  13-feature framework ยังคงอยู่ในไฟล์ 05"
    )
    print(
        "  แต่ behavior_dataset.csv จะเก็บเฉพาะ feature "
        "ที่ CIC สังเกตได้จริง"
    )
    print(
        "  Feature selection สำหรับ Model ยังทำใน Stage 08 จาก TRAIN เท่านั้น"
    )

    print()
    print("Saved:")
    print(
        f"  {OUTPUT_FILE.name}"
    )
    print(
        f"  {DESCRIPTIVE_FEATURE_FILE.name}"
    )
    print(
        f"  {CORRELATION_FILE.name}"
    )
    print(
        f"  {SUMMARY_FILE.name}"
    )
    print(
        f"  {MODEL_FEATURE_FILE.name}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
