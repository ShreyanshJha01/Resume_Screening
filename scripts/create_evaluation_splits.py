import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "resume_jd_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("CREATING LEAKAGE-SAFE EVALUATION SPLITS")
print("=" * 80)

print(f"\nLoading:")
print(FEATURE_FILE)

df = pd.read_csv(FEATURE_FILE)

print(
    f"\nDataset: "
    f"{len(df):,} rows × {len(df.columns)} columns"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_split(
    train_df,
    test_df,
    name
):

    train_file = (
        OUTPUT_DIR
        / f"{name}_train.csv"
    )

    test_file = (
        OUTPUT_DIR
        / f"{name}_test.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    print("\n" + "-" * 80)
    print(name.upper())

    print(
        f"Train rows : {len(train_df):,}"
    )

    print(
        f"Test rows  : {len(test_df):,}"
    )

    print(
        f"Train jobs : "
        f"{train_df['job_id'].nunique():,}"
    )

    print(
        f"Test jobs  : "
        f"{test_df['job_id'].nunique():,}"
    )

    print(
        f"Train people : "
        f"{train_df['person_id'].nunique():,}"
    )

    print(
        f"Test people  : "
        f"{test_df['person_id'].nunique():,}"
    )

    print(
        f"\nSaved:"
        f"\n  {train_file}"
        f"\n  {test_file}"
    )


# ============================================================
# SPLIT 1 — RANDOM ROW SPLIT
# ============================================================

print("\n" + "=" * 80)
print("1. RANDOM ROW SPLIT")
print("=" * 80)

train_random, test_random = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["pair_band"]
)

save_split(
    train_random,
    test_random,
    "random"
)


# ============================================================
# SPLIT 2 — UNSEEN CANDIDATES
# ============================================================

print("\n" + "=" * 80)
print("2. UNSEEN CANDIDATE SPLIT")
print("=" * 80)

people = (
    df["person_id"]
    .drop_duplicates()
    .to_numpy()
)

train_people, test_people = train_test_split(
    people,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_people = set(train_people)
test_people = set(test_people)

train_candidate = df[
    df["person_id"].isin(train_people)
].copy()

test_candidate = df[
    df["person_id"].isin(test_people)
].copy()

save_split(
    train_candidate,
    test_candidate,
    "candidate_holdout"
)


# ============================================================
# SPLIT 3 — UNSEEN JOBS
# ============================================================

print("\n" + "=" * 80)
print("3. UNSEEN JOB SPLIT")
print("=" * 80)

jobs = (
    df["job_id"]
    .drop_duplicates()
    .to_numpy()
)

train_jobs, test_jobs = train_test_split(
    jobs,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_jobs = set(train_jobs)
test_jobs = set(test_jobs)

train_job = df[
    df["job_id"].isin(train_jobs)
].copy()

test_job = df[
    df["job_id"].isin(test_jobs)
].copy()

save_split(
    train_job,
    test_job,
    "job_holdout"
)


# ============================================================
# SPLIT 4 — UNSEEN CANDIDATES + UNSEEN JOBS
# ============================================================

print("\n" + "=" * 80)
print("4. UNSEEN CANDIDATE + JOB SPLIT")
print("=" * 80)

# We intentionally use the candidate and job test sets
# created above.
#
# Training contains only rows where BOTH:
#   candidate is in training candidates
#   AND
#   job is in training jobs
#
# Testing contains only rows where BOTH:
#   candidate is in held-out candidates
#   AND
#   job is in held-out jobs

train_strict = df[
    df["person_id"].isin(train_people)
    &
    df["job_id"].isin(train_jobs)
].copy()

test_strict = df[
    df["person_id"].isin(test_people)
    &
    df["job_id"].isin(test_jobs)
].copy()

save_split(
    train_strict,
    test_strict,
    "candidate_job_holdout"
)


# ============================================================
# OVERLAP CHECKS
# ============================================================

print("\n" + "=" * 80)
print("5. LEAKAGE CHECKS")
print("=" * 80)


def check_overlap(
    train_df,
    test_df,
    name
):

    train_people = set(
        train_df["person_id"]
    )

    test_people = set(
        test_df["person_id"]
    )

    train_jobs = set(
        train_df["job_id"]
    )

    test_jobs = set(
        test_df["job_id"]
    )

    person_overlap = (
        train_people
        &
        test_people
    )

    job_overlap = (
        train_jobs
        &
        test_jobs
    )

    print(f"\n{name}")

    print(
        f"Person overlap: "
        f"{len(person_overlap):,}"
    )

    print(
        f"Job overlap: "
        f"{len(job_overlap):,}"
    )


check_overlap(
    train_candidate,
    test_candidate,
    "Candidate holdout"
)

check_overlap(
    train_job,
    test_job,
    "Job holdout"
)

check_overlap(
    train_strict,
    test_strict,
    "Candidate + Job holdout"
)


# ============================================================
# RELEVANCE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("6. RELEVANCE DISTRIBUTIONS")
print("=" * 80)


def print_distribution(
    df_split,
    name
):

    print(f"\n{name}")

    print(
        df_split["pair_band"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
        .to_string()
    )


print_distribution(
    train_random,
    "Random train"
)

print_distribution(
    test_random,
    "Random test"
)

print_distribution(
    train_candidate,
    "Candidate train"
)

print_distribution(
    test_candidate,
    "Candidate test"
)

print_distribution(
    train_job,
    "Job train"
)

print_distribution(
    test_job,
    "Job test"
)

print_distribution(
    train_strict,
    "Strict train"
)

print_distribution(
    test_strict,
    "Strict test"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SPLIT CREATION COMPLETE")
print("=" * 80)

print(
    f"\nEvaluation files created in:"
    f"\n{OUTPUT_DIR}"
)

for file in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    print(
        f"  {file.name}"
    )