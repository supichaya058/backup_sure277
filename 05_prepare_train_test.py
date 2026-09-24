from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_FILE = BASE_DIR / "selected_features.csv"
MONDAY_FILE = BASE_DIR / "Monday-WorkingHours.pcap_ISCX.csv"

CHUNK_SIZE = 50_000

TRAIN_SIZE = 150_000

TEST_BENIGN_PER_FILE = 20_000
TEST_ATTACK_PER_LABEL = 20_000

RANDOM_STATE = 42


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

features_df = pd.read_csv(FEATURE_FILE)

FEATURES = (
    features_df["feature"]
    .astype(str)
    .str.strip()
    .tolist()
)

REQUIRED_COLUMNS = FEATURES + ["Label"]


print("=" * 80)
print("PREPARE CIC-IDS2017 TRAIN / TEST")
print("=" * 80)

print(f"Selected features : {len(FEATURES)}")
print(f"Train size        : {TRAIN_SIZE:,}")
print()


# ============================================================
# HELPER: CLEAN CHUNK
# ============================================================

def clean_chunk(chunk):

    # Strip column names
    chunk.columns = (
        chunk.columns
        .astype(str)
        .str.strip()
    )

    # Check required columns
    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in chunk.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing_columns)
        )

    # Convert selected features to numeric
    for column in FEATURES:
        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )

    # Infinity -> NaN
    chunk[FEATURES] = chunk[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid rows
    chunk.dropna(
        subset=FEATURES,
        inplace=True
    )

    # Clean Label
    chunk["Label"] = (
        chunk["Label"]
        .astype(str)
        .str.strip()
    )

    return chunk


# ============================================================
# STEP 1
# MONDAY = TRAINING DATA
# ============================================================

print("=" * 80)
print("STEP 1: MONDAY TRAINING DATA")
print("=" * 80)

train_parts = []
train_rows = 0

for chunk in pd.read_csv(
    MONDAY_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk = clean_chunk(chunk)

    # Monday = BENIGN
    chunk = chunk[
        chunk["Label"].str.upper() == "BENIGN"
    ]

    if chunk.empty:
        continue

    remaining = TRAIN_SIZE - train_rows

    if remaining <= 0:
        break

    if len(chunk) > remaining:
        chunk = chunk.sample(
            n=remaining,
            random_state=RANDOM_STATE
        )

    train_parts.append(
        chunk[FEATURES].copy()
    )

    train_rows += len(chunk)

    print(
        f"Training rows collected: "
        f"{train_rows:,}/{TRAIN_SIZE:,}"
    )

    if train_rows >= TRAIN_SIZE:
        break


if not train_parts:
    raise RuntimeError(
        "ไม่สามารถสร้าง Training Dataset ได้"
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

train_df["target"] = 0

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


train_output = BASE_DIR / "train_cic.csv"

train_df.to_csv(
    train_output,
    index=False
)

print()
print(f"Train rows : {len(train_df):,}")
print(f"Train file : {train_output}")


# ============================================================
# STEP 2
# TEST DATA
# ============================================================

print()
print("=" * 80)
print("STEP 2: TEST DATA")
print("=" * 80)

test_parts = []

csv_files = sorted(BASE_DIR.glob("*.csv"))

excluded_files = {
    "selected_features.csv",
    "correlation_matrix.csv",
    "train_cic.csv",
    "test_cic.csv"
}


for file in csv_files:

    # Skip output/helper files
    if file.name in excluded_files:
        continue

    # Skip Monday
    if file.name == MONDAY_FILE.name:
        continue

    print()
    print("-" * 80)
    print(f"Processing: {file.name}")
    print("-" * 80)

    benign_parts = []

    attack_parts = {}

    benign_collected = 0
    attack_collected = {}

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk = clean_chunk(chunk)

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # BENIGN
        # ----------------------------------------------------

        benign = chunk[
            chunk["Label"].str.upper() == "BENIGN"
        ].copy()

        if not benign.empty:

            remaining = (
                TEST_BENIGN_PER_FILE
                - benign_collected
            )

            if remaining > 0:

                if len(benign) > remaining:
                    benign = benign.sample(
                        n=remaining,
                        random_state=RANDOM_STATE
                    )

                benign_parts.append(
                    benign[FEATURES].copy()
                )

                benign_collected += len(benign)

        # ----------------------------------------------------
        # ATTACK
        # ----------------------------------------------------

        attack = chunk[
            chunk["Label"].str.upper() != "BENIGN"
        ].copy()

        if not attack.empty:

            for attack_label, group in attack.groupby(
                "Label"
            ):

                current_count = attack_collected.get(
                    attack_label,
                    0
                )

                remaining = (
                    TEST_ATTACK_PER_LABEL
                    - current_count
                )

                if remaining <= 0:
                    continue

                if len(group) > remaining:
                    group = group.sample(
                        n=remaining,
                        random_state=RANDOM_STATE
                    )

                group = group.copy()

                group["attack_type"] = attack_label

                attack_parts.setdefault(
                    attack_label,
                    []
                ).append(
                    group[
                        FEATURES + ["attack_type"]
                    ]
                )

                attack_collected[attack_label] = (
                    current_count + len(group)
                )

    # --------------------------------------------------------
    # BENIGN DATAFRAME
    # --------------------------------------------------------

    if benign_parts:

        benign_df = pd.concat(
            benign_parts,
            ignore_index=True
        )

        benign_df["target"] = 0
        benign_df["attack_type"] = "BENIGN"

    else:

        benign_df = pd.DataFrame(
            columns=FEATURES + [
                "target",
                "attack_type"
            ]
        )

    # --------------------------------------------------------
    # ATTACK DATAFRAME
    # --------------------------------------------------------

    attack_frames = []

    for attack_label, parts in attack_parts.items():

        attack_df = pd.concat(
            parts,
            ignore_index=True
        )

        attack_df["target"] = 1

        attack_frames.append(
            attack_df[
                FEATURES +
                ["target", "attack_type"]
            ]
        )

    if attack_frames:

        attack_df = pd.concat(
            attack_frames,
            ignore_index=True
        )

    else:

        attack_df = pd.DataFrame(
            columns=FEATURES + [
                "target",
                "attack_type"
            ]
        )

    # --------------------------------------------------------
    # COMBINE FILE
    # --------------------------------------------------------

    file_test = pd.concat(
        [
            benign_df,
            attack_df
        ],
        ignore_index=True
    )

    test_parts.append(file_test)

    print(
        f"BENIGN : {len(benign_df):,}"
    )

    print(
        f"ATTACK : {len(attack_df):,}"
    )

    if attack_collected:

        print("\nAttack types:")

        for label, count in sorted(
            attack_collected.items()
        ):
            print(
                f"  {label:<35} {count:,}"
            )


# ============================================================
# COMBINE ALL TEST DATA
# ============================================================

if not test_parts:
    raise RuntimeError(
        "ไม่สามารถสร้าง Test Dataset ได้"
    )


test_df = pd.concat(
    test_parts,
    ignore_index=True
)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


test_output = BASE_DIR / "test_cic.csv"

test_df.to_csv(
    test_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL DATASET")
print("=" * 80)

print(
    f"Train rows : {len(train_df):,}"
)

print(
    f"Test rows  : {len(test_df):,}"
)

print()
print("Train target:")
print(
    train_df["target"].value_counts()
)

print()
print("Test target:")
print(
    test_df["target"].value_counts()
)

print()
print("Test attack types:")
print(
    test_df["attack_type"].value_counts()
)

print()
print(f"Saved train : {train_output}")
print(f"Saved test  : {test_output}")

print("=" * 80)
print("PREPARATION COMPLETE")
print("=" * 80)