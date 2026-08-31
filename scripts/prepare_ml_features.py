import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

STRUCTURED_FEATURES = [
    "experience_years",
    "num_skills",
    "num_experiences",
    "num_education_records",
    "resume_word_count",
    "resume_char_count",
    "num_required_skills",
    "job_word_count",
    "job_char_count",
]

MATCHING_FEATURES = [
    "num_matching_skills",
    "skill_match_ratio",
    "sbert_similarity",
]

ALL_NUMERIC_FEATURES = (
    STRUCTURED_FEATURES
    + MATCHING_FEATURES
)


# ============================================================
# LABEL
# ============================================================

LABEL_COLUMN = "pair_band"

LABEL_MAPPING = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


# ============================================================
# SPLITS
# ============================================================

SPLITS = [
    "random",
    "candidate_holdout",
    "job_holdout",
    "candidate_job_holdout",
]


# ============================================================
# LOAD + PREPARE
# ============================================================

print("=" * 80)
print("PREPARING ML FEATURE DATASETS")
print("=" * 80)

for split in SPLITS:

    train_file = (
        INPUT_DIR
        / f"{split}_train.csv"
    )

    test_file = (
        INPUT_DIR
        / f"{split}_test.csv"
    )

    train = pd.read_csv(
        train_file
    )

    test = pd.read_csv(
        test_file
    )

    print("\n" + "-" * 80)
    print(f"SPLIT: {split}")

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    train["label"] = (
        train[LABEL_COLUMN]
        .map(LABEL_MAPPING)
    )

    test["label"] = (
        test[LABEL_COLUMN]
        .map(LABEL_MAPPING)
    )

    # Safety check
    if train["label"].isna().any():
        raise ValueError(
            f"Unknown labels in {split} train."
        )

    if test["label"].isna().any():
        raise ValueError(
            f"Unknown labels in {split} test."
        )

    # --------------------------------------------------------
    # Check numeric columns
    # --------------------------------------------------------

    for col in ALL_NUMERIC_FEATURES:

        if col not in train.columns:
            raise ValueError(
                f"Missing feature: {col}"
            )

        if col not in test.columns:
            raise ValueError(
                f"Missing feature in test: {col}"
            )

    # --------------------------------------------------------
    # Create feature datasets
    # --------------------------------------------------------

    metadata_columns = [
        "person_id",
        "job_id",
        "pair_band",
        "label",
    ]

    # Keep metadata for ranking/evaluation.
    train_metadata = train[
        metadata_columns
    ].copy()

    test_metadata = test[
        metadata_columns
    ].copy()

    # --------------------------------------------------------
    # Structured features
    # --------------------------------------------------------

    train_structured = train[
        STRUCTURED_FEATURES
    ].copy()

    test_structured = test[
        STRUCTURED_FEATURES
    ].copy()

    train_structured["label"] = train["label"].astype(int)
    test_structured["label"] = test["label"].astype(int)

    # --------------------------------------------------------
    # Matching features
    # --------------------------------------------------------

    train_matching = train[
        ALL_NUMERIC_FEATURES
    ].copy()

    test_matching = test[
        ALL_NUMERIC_FEATURES
    ].copy()

    # Add target label to ML feature datasets
    # low = 0, medium = 1, high = 2
    train_matching["label"] = train["label"].astype(int)
    test_matching["label"] = test["label"].astype(int)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_structured_file = (
        OUTPUT_DIR
        / f"{split}_train_structured.csv"
    )

    test_structured_file = (
        OUTPUT_DIR
        / f"{split}_test_structured.csv"
    )

    train_matching_file = (
        OUTPUT_DIR
        / f"{split}_train_matching.csv"
    )

    test_matching_file = (
        OUTPUT_DIR
        / f"{split}_test_matching.csv"
    )

    train_metadata_file = (
        OUTPUT_DIR
        / f"{split}_train_metadata.csv"
    )

    test_metadata_file = (
        OUTPUT_DIR
        / f"{split}_test_metadata.csv"
    )

    train_structured.to_csv(
        train_structured_file,
        index=False
    )

    test_structured.to_csv(
        test_structured_file,
        index=False
    )

    train_matching.to_csv(
        train_matching_file,
        index=False
    )

    test_matching.to_csv(
        test_matching_file,
        index=False
    )

    train_metadata.to_csv(
        train_metadata_file,
        index=False
    )

    test_metadata.to_csv(
        test_metadata_file,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows : {len(test):,}"
    )

    print(
        f"Structured features: "
        f"{len(STRUCTURED_FEATURES)}"
    )

    print(
        f"Matching features: "
        f"{len(ALL_NUMERIC_FEATURES)}"
    )

    print(
        f"Train label distribution:"
    )

    print(
        train["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nTest label distribution:"
    )

    print(
        test["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ML FEATURE PREPARATION COMPLETE")
print("=" * 80)

print(
    f"\nOutput directory:"
    f"\n{OUTPUT_DIR}"
)

print("\nFeature sets:")

print(
    f"Structured: "
    f"{STRUCTURED_FEATURES}"
)

print(
    f"\nMatching:"
    f"\n{ALL_NUMERIC_FEATURES}"
)

print("\nLabel mapping:")

for name, value in LABEL_MAPPING.items():

    print(
        f"  {name} -> {value}"
    )