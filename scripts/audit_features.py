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
print("RESUME-JD FEATURE DATASET AUDIT")
print("=" * 80)

print(f"\nFeature file:")
print(FEATURE_FILE)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{FEATURE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

print("\nDataset loaded successfully.")

# ============================================================
# 1. SHAPE
# ============================================================

print("\n" + "=" * 80)
print("1. DATASET SHAPE")
print("=" * 80)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")

# ============================================================
# 2. COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("2. COLUMNS")
print("=" * 80)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:3}. {col}")

# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)

print(df.dtypes.to_string())

# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("4. MISSING VALUES")
print("=" * 80)

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": (missing / len(df) * 100).round(4)
})

print(missing_table.to_string())

# ============================================================
# 5. DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("5. DUPLICATE ROWS")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")

# ============================================================
# 6. EMPTY STRINGS
# ============================================================

print("\n" + "=" * 80)
print("6. EMPTY STRING CELLS")
print("=" * 80)

empty_strings = (
    df.astype("object")
      .apply(
          lambda col:
          col.astype(str).str.strip().eq("").sum()
      )
)

print(empty_strings.to_string())

# ============================================================
# 7. NUMERIC COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("7. NUMERIC COLUMNS")
print("=" * 80)

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

print(f"Number of numeric columns: {len(numeric_cols)}")

for col in numeric_cols:
    print(f"\n--- {col} ---")
    print(f"min    = {df[col].min()}")
    print(f"max    = {df[col].max()}")
    print(f"mean   = {df[col].mean()}")
    print(f"median = {df[col].median()}")

# ============================================================
# 8. ID COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("8. ID COLUMNS")
print("=" * 80)

for col in df.columns:

    col_lower = col.lower()

    if (
        col_lower == "job_id"
        or col_lower == "resume_id"
        or col_lower == "candidate_id"
        or col_lower.endswith("_id")
    ):

        print(
            f"{col}: "
            f"unique={df[col].nunique():,}, "
            f"nulls={df[col].isna().sum():,}"
        )

# ============================================================
# 9. JOB / RESUME PAIRS
# ============================================================

print("\n" + "=" * 80)
print("9. JOB-RESUME PAIR INFORMATION")
print("=" * 80)

if "job_id" in df.columns:
    print(
        f"Unique jobs: "
        f"{df['job_id'].nunique():,}"
    )

if "resume_id" in df.columns:
    print(
        f"Unique resumes: "
        f"{df['resume_id'].nunique():,}"
    )

if "candidate_id" in df.columns:
    print(
        f"Unique candidates: "
        f"{df['candidate_id'].nunique():,}"
    )

if (
    "job_id" in df.columns
    and "resume_id" in df.columns
):

    pairs = df[
        ["job_id", "resume_id"]
    ].drop_duplicates()

    print(
        f"Unique job-resume pairs: "
        f"{len(pairs):,}"
    )

    repeated_pairs = len(df) - len(pairs)

    print(
        f"Repeated job-resume rows: "
        f"{repeated_pairs:,}"
    )

# ============================================================
# 10. POSSIBLE TARGET / LABEL COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("10. POSSIBLE TARGET / LABEL COLUMNS")
print("=" * 80)

keywords = [
    "label",
    "target",
    "class",
    "score",
    "relevance",
    "match",
    "similarity",
    "rating"
]

possible_targets = []

for col in df.columns:

    col_lower = col.lower()

    if any(
        keyword in col_lower
        for keyword in keywords
    ):
        possible_targets.append(col)

if possible_targets:

    for col in possible_targets:

        print(f"\n--- {col} ---")

        print(
            f"dtype: {df[col].dtype}"
        )

        print(
            f"unique values: "
            f"{df[col].nunique():,}"
        )

        if df[col].nunique() <= 20:

            print("value counts:")

            print(
                df[col]
                .value_counts(dropna=False)
                .to_string()
            )

else:

    print(
        "No obvious target/label column detected."
    )

# ============================================================
# 11. SAMPLE ROWS
# ============================================================

print("\n" + "=" * 80)
print("11. SAMPLE ROWS")
print("=" * 80)

print(
    df.head(5).to_string(
        index=False,
        max_cols=50
    )
)

# ============================================================
# 12. SAVE AUDIT REPORT
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    REPORT_DIR
    / "feature_dataset_audit.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESUME-JD FEATURE DATASET AUDIT\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Rows: {df.shape[0]:,}\n"
    )

    f.write(
        f"Columns: {df.shape[1]:,}\n\n"
    )

    f.write("COLUMNS\n")
    f.write("-" * 80 + "\n")

    for col in df.columns:
        f.write(f"{col}\n")

    f.write("\nDATA TYPES\n")
    f.write("-" * 80 + "\n")

    f.write(
        df.dtypes.to_string()
    )

    f.write("\n\nMISSING VALUES\n")
    f.write("-" * 80 + "\n")

    f.write(
        missing_table.to_string()
    )

    f.write("\n\nDUPLICATE ROWS\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"{duplicate_count:,}\n"
    )

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(
    f"\nAudit report saved to:\n"
    f"{REPORT_FILE}"
)