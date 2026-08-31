from pathlib import Path
import pandas as pd

from text_cleaning import clean_job_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def prepare_job_data():
    """
    Prepare job descriptions for the feature-engineering stage.
    """

    input_path = RAW_DATA_DIR / "job_descriptions.csv"

    jobs = pd.read_csv(input_path)

    print("Job dataset loaded.")
    print("Shape:", jobs.shape)

    # ---------------------------------------------------------
    # 1. Handle missing values
    # ---------------------------------------------------------

    text_columns = [
        "title",
        "category",
        "description",
        "required_skills",
        "job_text"
    ]

    for column in text_columns:
        if column in jobs.columns:
            jobs[column] = jobs[column].fillna("")

    # ---------------------------------------------------------
    # 2. Clean job text
    # ---------------------------------------------------------

    jobs["job_text_clean"] = (
        jobs["job_text"]
        .apply(clean_job_text)
    )

    # ---------------------------------------------------------
    # 3. Remove exact duplicate jobs
    # ---------------------------------------------------------

    before = len(jobs)

    jobs = (
        jobs
        .drop_duplicates(
            subset=[
                "title",
                "category",
                "description",
                "required_skills",
                "job_text"
            ],
            keep="first"
        )
        .reset_index(drop=True)
    )

    after = len(jobs)

    print("Jobs before duplicate removal:", before)
    print("Duplicate jobs removed:", before - after)
    print("Jobs after duplicate removal:", after)

    # ---------------------------------------------------------
    # 4. Save preprocessed jobs
    # ---------------------------------------------------------

    INTERIM_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        INTERIM_DATA_DIR
        / "preprocessed_jobs.csv"
    )

    jobs.to_csv(
        output_path,
        index=False
    )

    print("\nJob preprocessing completed.")
    print("Shape:", jobs.shape)
    print("Columns:", jobs.columns.tolist())
    print("Saved to:", output_path)

    return jobs


if __name__ == "__main__":
    prepare_job_data()