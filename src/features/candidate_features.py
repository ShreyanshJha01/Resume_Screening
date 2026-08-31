from pathlib import Path
import re

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Experience date parsing
# ---------------------------------------------------------

def parse_experience_date(value):
    """
    Convert the mixed date formats in the experience dataset
    into pandas timestamps.

    Supported formats:
    - Present
    - MM/YYYY
    - MM/YY
    - YYYY
    """

    value = str(value).strip()

    if value.lower() == "present":
        return pd.Timestamp.today()

    # MM/YYYY
    if re.fullmatch(r"\d{2}/\d{4}", value):
        month, year = value.split("/")
        month = int(month)

        if 1 <= month <= 12:
            return pd.Timestamp(
                year=int(year),
                month=month,
                day=1
            )

        return pd.NaT

    # MM/YY
    if re.fullmatch(r"\d{2}/\d{2}", value):
        month, year = value.split("/")
        month = int(month)

        if 1 <= month <= 12:
            return pd.to_datetime(
                f"{month:02d}/{year}",
                format="%m/%y"
            )

        return pd.NaT

    # YYYY
    if re.fullmatch(r"\d{4}", value):
        return pd.to_datetime(
            value,
            format="%Y",
            errors="coerce"
        )

    return pd.NaT


# ---------------------------------------------------------
# Candidate skill normalization
# ---------------------------------------------------------

def normalize_candidate_skill(skill):
    """
    Normalize an individual candidate skill.
    """

    skill = str(skill).lower().strip()

    # Remove experience duration such as:
    # Python (3 years)
    skill = re.sub(
        r"\s*\(\s*\d+(?:\.\d+)?\s*years?\s*\)",
        "",
        skill
    )

    skill = re.sub(
        r"\s+",
        " ",
        skill
    ).strip()

    return skill


def normalize_candidate_skill_set(text):
    """
    Convert pipe-separated candidate skills
    into a normalized Python set.
    """

    skills = str(text).split("|")

    return {
        normalize_candidate_skill(skill)
        for skill in skills
        if str(skill).strip()
    }


# ---------------------------------------------------------
# Main feature engineering
# ---------------------------------------------------------

def create_candidate_features():
    """
    Create candidate-level features from the preprocessed
    candidate dataset and raw experience data.
    """

    # -----------------------------------------------------
    # 1. Load preprocessed candidates
    # -----------------------------------------------------

    candidates_path = (
        INTERIM_DATA_DIR /
        "preprocessed_candidates.csv"
    )

    cleaned_df = pd.read_csv(candidates_path)

    print("Preprocessed candidate dataset loaded.")
    print("Shape:", cleaned_df.shape)

    # -----------------------------------------------------
    # 2. Resume text-length features
    # -----------------------------------------------------

    cleaned_df["resume_word_count"] = (
        cleaned_df["resume_text_clean"]
        .str.split()
        .str.len()
    )

    cleaned_df["resume_char_count"] = (
        cleaned_df["resume_text_clean"]
        .str.len()
    )

    # -----------------------------------------------------
    # 3. Number of skills
    # -----------------------------------------------------

    cleaned_df["num_skills"] = (
        cleaned_df["skills_text"]
        .apply(
            lambda x: len([
                skill.strip()
                for skill in str(x).split("|")
                if skill.strip()
            ])
        )
    )

    # -----------------------------------------------------
    # 4. Number of experience records
    # -----------------------------------------------------

    cleaned_df["num_experiences"] = (
        cleaned_df["experience_text"]
        .apply(
            lambda x: len([
                item.strip()
                for item in str(x).split("||")
                if item.strip()
            ])
        )
    )

    # -----------------------------------------------------
    # 5. Number of education records
    # -----------------------------------------------------

    cleaned_df["num_education_records"] = (
        cleaned_df["education_text"]
        .apply(
            lambda x: len([
                item.strip()
                for item in str(x).split("||")
                if item.strip()
            ])
        )
    )

    # -----------------------------------------------------
    # 6. Load raw experience data
    # -----------------------------------------------------

    experience_path = (
        RAW_DATA_DIR /
        "04_experience.csv"
    )

    experience = pd.read_csv(experience_path)

    print("Raw experience dataset loaded.")
    print("Shape:", experience.shape)

    # -----------------------------------------------------
    # 7. Parse experience dates
    # -----------------------------------------------------

    experience_dates = experience.copy()

    experience_dates["start_parsed"] = (
        experience_dates["start_date"]
        .apply(parse_experience_date)
    )

    experience_dates["end_parsed"] = (
        experience_dates["end_date"]
        .apply(parse_experience_date)
    )

    # -----------------------------------------------------
    # 8. Calculate career span
    # -----------------------------------------------------

    career_span = (
        experience_dates
        .groupby("person_id")
        .agg(
            earliest_start=("start_parsed", "min"),
            latest_end=("end_parsed", "max")
        )
        .reset_index()
    )

    career_span["experience_years"] = (
        (
            career_span["latest_end"]
            - career_span["earliest_start"]
        ).dt.days / 365.25
    )

    career_span["experience_years"] = (
        career_span["experience_years"]
        .clip(lower=0)
        .fillna(0)
    )

    # -----------------------------------------------------
    # 9. Merge experience years into candidate dataset
    # -----------------------------------------------------

    cleaned_df = cleaned_df.drop(
        columns=["experience_years"],
        errors="ignore"
    )

    cleaned_df = cleaned_df.merge(
        career_span[
            [
                "person_id",
                "experience_years"
            ]
        ],
        on="person_id",
        how="left"
    )

    cleaned_df["experience_years"] = (
        cleaned_df["experience_years"]
        .fillna(0)
    )

    # -----------------------------------------------------
    # 10. Normalize candidate skills
    # -----------------------------------------------------

    cleaned_df["skill_set"] = (
        cleaned_df["skills_text"]
        .apply(normalize_candidate_skill_set)
    )

    # -----------------------------------------------------
    # 11. Save candidate feature dataset
    # -----------------------------------------------------

    output_path = (
        INTERIM_DATA_DIR /
        "candidate_features.csv"
    )

    # Convert Python sets to strings for CSV storage
    cleaned_df["skill_set"] = (
        cleaned_df["skill_set"]
        .apply(lambda x: " | ".join(sorted(x)))
    )

    cleaned_df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # 12. Display results
    # -----------------------------------------------------

    print("\nCandidate feature engineering completed.")

    print("Final shape:", cleaned_df.shape)

    print("\nCreated candidate features:")
    print([
        "resume_word_count",
        "resume_char_count",
        "num_skills",
        "num_experiences",
        "num_education_records",
        "experience_years",
        "skill_set"
    ])

    print("\nExperience statistics:")
    print(
        cleaned_df["experience_years"].describe()
    )

    print("\nSaved to:")
    print(output_path)

    return cleaned_df


if __name__ == "__main__":
    create_candidate_features()