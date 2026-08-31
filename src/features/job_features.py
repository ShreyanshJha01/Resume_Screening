from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def create_job_features():
    """
    Create job-level features from the preprocessed job dataset.
    """

    # ---------------------------------------------------------
    # 1. Load preprocessed jobs
    # ---------------------------------------------------------

    input_path = (
        INTERIM_DATA_DIR /
        "preprocessed_jobs.csv"
    )

    jobs = pd.read_csv(input_path)

    print("Preprocessed job dataset loaded.")
    print("Shape:", jobs.shape)

    # ---------------------------------------------------------
    # 2. Number of required skills
    # ---------------------------------------------------------

    jobs["num_required_skills"] = (
        jobs["required_skills"]
        .fillna("")
        .apply(
            lambda x: len([
                skill.strip()
                for skill in str(x).split(",")
                if skill.strip()
            ])
        )
    )

    # ---------------------------------------------------------
    # 3. Job word count
    # ---------------------------------------------------------

    jobs["job_word_count"] = (
        jobs["job_text_clean"]
        .fillna("")
        .str.split()
        .str.len()
    )

    # ---------------------------------------------------------
    # 4. Job character count
    # ---------------------------------------------------------

    jobs["job_char_count"] = (
        jobs["job_text_clean"]
        .fillna("")
        .str.len()
    )

    # ---------------------------------------------------------
    # 5. Save job feature dataset
    # ---------------------------------------------------------

    output_path = (
        INTERIM_DATA_DIR /
        "job_features.csv"
    )

    jobs.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # 6. Display results
    # ---------------------------------------------------------

    print("\nJob feature engineering completed.")
    print("Final shape:", jobs.shape)

    print("\nCreated job features:")
    print([
        "num_required_skills",
        "job_word_count",
        "job_char_count"
    ])

    print("\nFeature summary:")
    print(
        jobs[
            [
                "num_required_skills",
                "job_word_count",
                "job_char_count"
            ]
        ].describe()
    )

    print("\nSaved to:")
    print(output_path)

    return jobs


if __name__ == "__main__":
    create_job_features()