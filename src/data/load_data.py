from pathlib import Path
import pandas as pd


# Project root: capstone/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw_data():
    """Load all original datasets from data/raw/."""

    people = pd.read_csv(RAW_DATA_DIR / "01_people.csv")
    abilities = pd.read_csv(RAW_DATA_DIR / "02_abilities.csv")
    education = pd.read_csv(RAW_DATA_DIR / "03_education.csv")
    experience = pd.read_csv(RAW_DATA_DIR / "04_experience.csv")
    person_skills = pd.read_csv(RAW_DATA_DIR / "05_person_skills.csv")
    skills = pd.read_csv(RAW_DATA_DIR / "06_skills.csv")
    job_descriptions = pd.read_csv(RAW_DATA_DIR / "job_descriptions.csv")

    return {
        "people": people,
        "abilities": abilities,
        "education": education,
        "experience": experience,
        "person_skills": person_skills,
        "skills": skills,
        "job_descriptions": job_descriptions,
    }


if __name__ == "__main__":
    data = load_raw_data()

    print("\nRaw datasets loaded successfully:\n")

    for name, df in data.items():
        print(f"{name:20} {df.shape}")