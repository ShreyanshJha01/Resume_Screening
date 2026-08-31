from pathlib import Path
import ast
import random

import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42

HIGH_SAMPLES = 5
MEDIUM_SAMPLES = 5
LOW_SAMPLES = 10

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def parse_skill_set(value):
    """
    Convert the skill_set stored in CSV back into a Python set.
    """

    if pd.isna(value):
        return set()

    if isinstance(value, set):
        return value

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (set, list, tuple)):
            return {
                str(skill).strip().lower()
                for skill in parsed
                if str(skill).strip()
            }

    except (ValueError, SyntaxError):
        pass

    # Fallback for pipe-separated values
    return {
        skill.strip().lower()
        for skill in str(value).split("|")
        if skill.strip()
    }


def parse_required_skills(value):
    """
    Convert required_skills into a normalized Python set.
    """

    if pd.isna(value):
        return set()

    return {
        skill.strip().lower()
        for skill in str(value).split(",")
        if skill.strip()
    }


def calculate_skill_match(candidate_skills, required_skills):
    """
    Calculate matching skills, missing skills and skill ratio.
    """

    matching = candidate_skills.intersection(required_skills)
    missing = required_skills.difference(candidate_skills)

    if len(required_skills) == 0:
        ratio = 0.0
    else:
        ratio = len(matching) / len(required_skills)

    return matching, missing, ratio


def get_pair_band(skill_ratio):
    """
    Assign the same type of matching band used in the
    previous project logic.
    """

    if skill_ratio >= 0.50:
        return "high"

    elif skill_ratio >= 0.25:
        return "medium"

    return "low"


# =========================================================
# MAIN FUNCTION
# =========================================================

def create_matching_features():

    random.seed(RANDOM_SEED)

    # -----------------------------------------------------
    # 1. Load candidate features
    # -----------------------------------------------------

    candidate_path = (
        INTERIM_DATA_DIR /
        "candidate_features.csv"
    )

    candidates = pd.read_csv(candidate_path)

    print("Candidate feature dataset loaded.")
    print("Shape:", candidates.shape)

    # -----------------------------------------------------
    # 2. Load job features
    # -----------------------------------------------------

    job_path = (
        INTERIM_DATA_DIR /
        "job_features.csv"
    )

    jobs = pd.read_csv(job_path)

    print("\nJob feature dataset loaded.")
    print("Shape:", jobs.shape)

    # -----------------------------------------------------
    # 3. Convert skill columns to sets
    # -----------------------------------------------------

    candidates["skill_set"] = (
        candidates["skill_set"]
        .apply(parse_skill_set)
    )

    jobs["required_skill_set"] = (
        jobs["required_skills"]
        .apply(parse_required_skills)
    )

    # -----------------------------------------------------
    # 4. Create candidate-job pairs
    #
    # For every job:
    #   5 high
    #   5 medium
    #   10 low
    #
    # Maximum = 20 candidates per job
    # -----------------------------------------------------

    candidate_records = []

    candidate_indices = list(candidates.index)

    print("\nCreating balanced candidate-job pairs...")

    for job_index, job in jobs.iterrows():

        required_skills = job["required_skill_set"]

        high_candidates = []
        medium_candidates = []
        low_candidates = []

        # Shuffle candidate order for reproducibility
        shuffled_indices = candidate_indices.copy()
        random.shuffle(shuffled_indices)

        for candidate_index in shuffled_indices:

            candidate = candidates.loc[candidate_index]

            candidate_skills = candidate["skill_set"]

            _, _, ratio = calculate_skill_match(
                candidate_skills,
                required_skills
            )

            band = get_pair_band(ratio)

            if band == "high":
                if len(high_candidates) < HIGH_SAMPLES:
                    high_candidates.append(candidate_index)

            elif band == "medium":
                if len(medium_candidates) < MEDIUM_SAMPLES:
                    medium_candidates.append(candidate_index)

            else:
                if len(low_candidates) < LOW_SAMPLES:
                    low_candidates.append(candidate_index)

            # Stop once all required samples are collected
            if (
                len(high_candidates) == HIGH_SAMPLES
                and
                len(medium_candidates) == MEDIUM_SAMPLES
                and
                len(low_candidates) == LOW_SAMPLES
            ):
                break

        selected = (
            [(idx, "high") for idx in high_candidates]
            + [(idx, "medium") for idx in medium_candidates]
            + [(idx, "low") for idx in low_candidates]
        )

        for candidate_index, band in selected:

            candidate_records.append(
                {
                    "candidate_index": candidate_index,
                    "job_index": job_index,
                    "pair_band": band
                }
            )

    pairs = pd.DataFrame(candidate_records)

    print(
        "Candidate-job pairs created:",
        len(pairs)
    )

    # -----------------------------------------------------
    # 5. Merge candidate and job information
    # -----------------------------------------------------

    candidate_data = candidates.reset_index(
        names="candidate_index"
    )

    job_data = jobs.reset_index(
        names="job_index"
    )

    matching_df = pairs.merge(
        candidate_data,
        on="candidate_index",
        how="left"
    )

    matching_df = matching_df.merge(
        job_data,
        on="job_index",
        how="left",
        suffixes=("", "_job")
    )

    # -----------------------------------------------------
    # 6. Calculate skill matching
    # -----------------------------------------------------

    matching_skills = []
    missing_skills = []
    num_matching_skills = []
    skill_match_ratio = []

    for _, row in matching_df.iterrows():

        candidate_skills = row["skill_set"]
        required_skills = row["required_skill_set"]

        matched, missing, ratio = calculate_skill_match(
            candidate_skills,
            required_skills
        )

        matching_skills.append(
            " | ".join(sorted(matched))
        )

        missing_skills.append(
            " | ".join(sorted(missing))
        )

        num_matching_skills.append(
            len(matched)
        )

        skill_match_ratio.append(
            ratio
        )

    matching_df["matching_skills"] = matching_skills
    matching_df["missing_skills"] = missing_skills
    matching_df["num_matching_skills"] = num_matching_skills
    matching_df["skill_match_ratio"] = skill_match_ratio

    # -----------------------------------------------------
    # 7. SBERT semantic similarity
    # -----------------------------------------------------

    print("\nLoading SBERT model...")
    print("Model:", SBERT_MODEL_NAME)

    model = SentenceTransformer(SBERT_MODEL_NAME)

    resume_texts = (
        matching_df["resume_text_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    job_texts = (
        matching_df["job_text_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("Generating resume embeddings...")

    resume_embeddings = model.encode(
        resume_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Generating job embeddings...")

    job_embeddings = model.encode(
        job_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Since embeddings are normalized, dot product = cosine similarity
    similarities = (
        (resume_embeddings * job_embeddings)
        .sum(axis=1)
    )

    matching_df["sbert_similarity"] = similarities

    # -----------------------------------------------------
    # 8. Combined matching score
    # -----------------------------------------------------

    matching_df["combined_score"] = (
        0.50 * matching_df["skill_match_ratio"]
        +
        0.50 * matching_df["sbert_similarity"]
    )

    # -----------------------------------------------------
    # 9. Create relevance label
    #
    # 0 = Low relevance
    # 1 = Medium relevance
    # 2 = High relevance
    # -----------------------------------------------------

    def create_relevance_label(score):

        if score < 0.30:
            return 0

        elif score <= 0.50:
            return 1

        return 2

    matching_df["relevance_label"] = (
        matching_df["combined_score"]
        .apply(create_relevance_label)
    )

    # -----------------------------------------------------
    # 10. Remove helper columns
    # -----------------------------------------------------

    matching_df.drop(
        columns=[
            "candidate_index",
            "job_index",
            "required_skill_set"
        ],
        inplace=True,
        errors="ignore"
    )

    # -----------------------------------------------------
    # 11. Save final matching dataset
    # -----------------------------------------------------

    output_path = (
        PROCESSED_DATA_DIR /
        "resume_jd_matching_dataset.csv"
    )

    matching_df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # 12. Display results
    # -----------------------------------------------------

    print("\nMatching feature engineering completed.")

    print(
        "Final shape:",
        matching_df.shape
    )

    print("\nMatching features created:")

    print([
        "matching_skills",
        "missing_skills",
        "num_matching_skills",
        "skill_match_ratio",
        "sbert_similarity",
        "combined_score",
        "relevance_label"
    ])

    print("\nPair-band distribution:")
    print(
        matching_df["pair_band"]
        .value_counts()
    )

    print("\nRelevance-label distribution:")
    print(
        matching_df["relevance_label"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved to:")
    print(output_path)

    return matching_df


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":
    create_matching_features()