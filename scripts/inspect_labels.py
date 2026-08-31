import pandas as pd
from pathlib import Path

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

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LABEL / SCORE INSPECTION")
print("=" * 80)

df = pd.read_csv(FEATURE_FILE)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

# ============================================================
# 1. PAIR BAND
# ============================================================

print("\n" + "=" * 80)
print("1. PAIR_BAND")
print("=" * 80)

print(df["pair_band"].value_counts(dropna=False))

print("\nPercentages:")

print(
    (
        df["pair_band"]
        .value_counts(normalize=True, dropna=False)
        * 100
    ).round(2)
)

# ============================================================
# 2. SKILL MATCH BAND
# ============================================================

print("\n" + "=" * 80)
print("2. SKILL_MATCH_BAND")
print("=" * 80)

print(
    df["skill_match_band"]
    .value_counts(dropna=False)
)

print("\nPercentages:")

print(
    (
        df["skill_match_band"]
        .value_counts(normalize=True, dropna=False)
        * 100
    ).round(2)
)

# ============================================================
# 3. COMBINED SCORE
# ============================================================

print("\n" + "=" * 80)
print("3. COMBINED_SCORE")
print("=" * 80)

print(
    df["combined_score"].describe()
)

# ============================================================
# 4. SKILL MATCH RATIO
# ============================================================

print("\n" + "=" * 80)
print("4. SKILL_MATCH_RATIO")
print("=" * 80)

print(
    df["skill_match_ratio"].describe()
)

# ============================================================
# 5. SBERT SIMILARITY
# ============================================================

print("\n" + "=" * 80)
print("5. SBERT_SIMILARITY")
print("=" * 80)

print(
    df["sbert_similarity"].describe()
)

# ============================================================
# 6. PAIR BAND × SCORE
# ============================================================

print("\n" + "=" * 80)
print("6. PAIR_BAND vs COMBINED_SCORE")
print("=" * 80)

print(
    df.groupby("pair_band")["combined_score"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max"
    )
    .round(4)
    .to_string()
)

# ============================================================
# 7. PAIR BAND × SKILL MATCH
# ============================================================

print("\n" + "=" * 80)
print("7. PAIR_BAND vs SKILL_MATCH_RATIO")
print("=" * 80)

print(
    df.groupby("pair_band")["skill_match_ratio"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max"
    )
    .round(4)
    .to_string()
)

# ============================================================
# 8. PAIR BAND × SBERT
# ============================================================

print("\n" + "=" * 80)
print("8. PAIR_BAND vs SBERT_SIMILARITY")
print("=" * 80)

print(
    df.groupby("pair_band")["sbert_similarity"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max"
    )
    .round(4)
    .to_string()
)

# ============================================================
# 9. PAIR BAND × SKILL BAND
# ============================================================

print("\n" + "=" * 80)
print("9. PAIR_BAND × SKILL_MATCH_BAND")
print("=" * 80)

print(
    pd.crosstab(
        df["pair_band"],
        df["skill_match_band"],
        normalize="index"
    )
    .round(4)
    .to_string()
)

# ============================================================
# 10. CORRELATIONS
# ============================================================

print("\n" + "=" * 80)
print("10. CORRELATIONS")
print("=" * 80)

numeric_columns = [
    "experience_years",
    "num_skills",
    "num_experiences",
    "num_education_records",
    "resume_word_count",
    "resume_char_count",
    "num_required_skills",
    "job_word_count",
    "job_char_count",
    "num_matching_skills",
    "skill_match_ratio",
    "sbert_similarity",
    "combined_score",
]

existing_numeric = [
    col for col in numeric_columns
    if col in df.columns
]

print(
    df[existing_numeric]
    .corr(numeric_only=True)["combined_score"]
    .sort_values(ascending=False)
    .round(4)
    .to_string()
)

# ============================================================
# 11. PER-JOB DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("11. ROWS PER JOB")
print("=" * 80)

rows_per_job = df.groupby("job_id").size()

print(
    rows_per_job.describe()
    .round(2)
)

print(
    f"\nJobs: {rows_per_job.shape[0]:,}"
)

print(
    f"Minimum candidates/pairs per job: "
    f"{rows_per_job.min():,}"
)

print(
    f"Maximum candidates/pairs per job: "
    f"{rows_per_job.max():,}"
)

# ============================================================
# 12. ROWS PER PERSON
# ============================================================

print("\n" + "=" * 80)
print("12. ROWS PER PERSON")
print("=" * 80)

rows_per_person = df.groupby("person_id").size()

print(
    rows_per_person.describe()
    .round(2)
)

print(
    f"\nPeople: {rows_per_person.shape[0]:,}"
)

print(
    f"Minimum jobs per person: "
    f"{rows_per_person.min():,}"
)

print(
    f"Maximum jobs per person: "
    f"{rows_per_person.max():,}"
)

# ============================================================
# 13. SAMPLE RECORDS BY PAIR BAND
# ============================================================

print("\n" + "=" * 80)
print("13. SAMPLE RECORDS")
print("=" * 80)

for band in df["pair_band"].dropna().unique():

    print(f"\n--- PAIR BAND: {band} ---")

    sample = df[
        df["pair_band"] == band
    ][
        [
            "person_id",
            "job_id",
            "pair_band",
            "skill_match_ratio",
            "sbert_similarity",
            "combined_score",
            "skill_match_band"
        ]
    ].head(5)

    print(
        sample.to_string(index=False)
    )

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)