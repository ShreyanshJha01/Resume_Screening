from pathlib import Path
import pandas as pd

from text_cleaning import clean_resume_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def prepare_candidate_data():
    """
    Preprocess the candidate profile dataset.
    """

    input_path = INTERIM_DATA_DIR / "candidate_profiles.csv"

    resume_df = pd.read_csv(input_path)

    print("Candidate dataset loaded.")
    print("Shape:", resume_df.shape)

    # ---------------------------------------------------------
    # 1. Handle missing values
    # ---------------------------------------------------------

    text_columns = [
        "ability",
        "education_text",
        "experience_text",
        "skills_text",
        "resume_text"
    ]

    resume_df[text_columns] = (
        resume_df[text_columns].fillna("")
    )

    # ---------------------------------------------------------
    # 2. Create resume group
    # ---------------------------------------------------------

    resume_df["resume_group"] = (
        resume_df["resume_text"]
        .astype(str)
        .str.strip()
        .factorize()[0]
    )

    # ---------------------------------------------------------
    # 3. Clean resume text
    # ---------------------------------------------------------

    resume_df["resume_text_clean"] = (
        resume_df["resume_text"]
        .apply(clean_resume_text)
    )

    # ---------------------------------------------------------
    # 4. Remove exact duplicate resume records
    # ---------------------------------------------------------

    before = len(resume_df)

    resume_df = (
        resume_df
        .drop_duplicates(
            subset=[
                "ability",
                "education_text",
                "experience_text",
                "skills_text",
                "resume_text"
            ],
            keep="first"
        )
        .reset_index(drop=True)
    )

    after = len(resume_df)

    print("Candidates before duplicate removal:", before)
    print("Duplicate records removed:", before - after)
    print("Candidates after duplicate removal:", after)


    # ---------------------------------------------------------
    # 5. Save preprocessed candidate dataset
    # ---------------------------------------------------------

    output_path = (
        INTERIM_DATA_DIR
        / "preprocessed_candidates.csv"
    )


    resume_df.to_csv(
        output_path,
        index=False
    )

    print("\nCandidate preprocessing completed.")
    print("Shape:", resume_df.shape)
    print("Saved to:", output_path)

    return resume_df


if __name__ == "__main__":
    prepare_candidate_data()