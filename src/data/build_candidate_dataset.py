from pathlib import Path
import pandas as pd

from load_data import load_raw_data


# Project root: capstone/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Output directory
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def build_candidate_profiles():
    """
    Build one consolidated profile per candidate by aggregating
    abilities, education, experience, and skills before merging.
    """

    data = load_raw_data()

    people = data["people"]
    abilities = data["abilities"]
    education = data["education"]
    experience = data["experience"]
    person_skills = data["person_skills"]

    # ---------------------------------------------------------
    # 1. Aggregate abilities
    # ---------------------------------------------------------
    abilities_agg = (
        abilities
        .groupby("person_id")["ability"]
        .apply(
            lambda x: " | ".join(
                x.dropna().astype(str).unique()
            )
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 2. Aggregate education
    # ---------------------------------------------------------
    education_agg = (
        education
        .fillna("")
        .astype(str)
        .assign(
            education_text=lambda df:
                df["program"] + " at "
                + df["institution"]
                + " | Location: "
                + df["location"]
        )
        .groupby("person_id")["education_text"]
        .apply(
            lambda x: " || ".join(x.unique())
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3. Aggregate experience
    # ---------------------------------------------------------
    experience_agg = (
        experience
        .fillna("")
        .astype(str)
        .assign(
            experience_text=lambda df:
                df["title"] + " at "
                + df["firm"]
                + " | Location: "
                + df["location"]
                + " | From: "
                + df["start_date"]
                + " | To: "
                + df["end_date"]
        )
        .groupby("person_id")["experience_text"]
        .apply(
            lambda x: " || ".join(x.unique())
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 4. Aggregate skills
    # ---------------------------------------------------------
    skills_agg = (
        person_skills
        .groupby("person_id")["skill"]
        .apply(
            lambda x: " | ".join(
                x.dropna().astype(str).unique()
            )
        )
        .reset_index()
        .rename(columns={"skill": "skills_text"})
    )

    # ---------------------------------------------------------
    # 5. Make sure person_id has the same type
    # ---------------------------------------------------------
    id_columns = [
        people,
        abilities_agg,
        education_agg,
        experience_agg,
        skills_agg,
    ]

    for df in id_columns:
        df["person_id"] = pd.to_numeric(
            df["person_id"],
            errors="coerce"
        ).astype("Int64")

    # ---------------------------------------------------------
    # 6. Merge aggregated datasets
    # ---------------------------------------------------------
    resume_df = (
        people
        .merge(
            abilities_agg,
            on="person_id",
            how="left"
        )
        .merge(
            education_agg,
            on="person_id",
            how="left"
        )
        .merge(
            experience_agg,
            on="person_id",
            how="left"
        )
        .merge(
            skills_agg,
            on="person_id",
            how="left"
        )
    )

    # ---------------------------------------------------------
    # 7. Create combined resume text
    # ---------------------------------------------------------
    resume_df["resume_text"] = (
        "Abilities: "
        + resume_df["ability"].fillna("")
        + " | Education: "
        + resume_df["education_text"].fillna("")
        + " | Experience: "
        + resume_df["experience_text"].fillna("")
        + " | Skills: "
        + resume_df["skills_text"].fillna("")
    )

    # ---------------------------------------------------------
    # 8. Remove personally identifying information
    # ---------------------------------------------------------
    resume_df = resume_df.drop(
        columns=[
            "name",
            "email",
            "phone",
            "linkedin",
        ],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # 9. Keep only required candidate-profile columns
    # ---------------------------------------------------------
    resume_df = resume_df[
        [
            "person_id",
            "ability",
            "education_text",
            "experience_text",
            "skills_text",
            "resume_text",
        ]
    ]

    # ---------------------------------------------------------
    # 10. Fill missing text fields
    # ---------------------------------------------------------
    text_columns = [
        "ability",
        "education_text",
        "experience_text",
        "skills_text",
        "resume_text",
    ]

    resume_df[text_columns] = resume_df[text_columns].fillna("")

    # ---------------------------------------------------------
    # 11. Save intermediate dataset
    # ---------------------------------------------------------
    INTERIM_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        INTERIM_DATA_DIR
        / "candidate_profiles.csv"
    )

    resume_df.to_csv(
        output_path,
        index=False
    )

    return resume_df


if __name__ == "__main__":
    resume_df = build_candidate_profiles()

    print("\nCandidate profile dataset created successfully!")
    print(f"Shape: {resume_df.shape}")

    print("\nColumns:")
    print(resume_df.columns.tolist())

    print("\nDuplicate person IDs:")
    print(resume_df["person_id"].duplicated().sum())

    print(f"\nSaved to:")
    print(
        INTERIM_DATA_DIR
        / "candidate_profiles.csv"
    )