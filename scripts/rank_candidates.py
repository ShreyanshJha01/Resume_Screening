from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data" / "processed" / "resume_jd_features.csv"
MODEL_DIR = ROOT / "models"

DEFAULT_MODEL = "lightgbm"

LABEL_NAMES = {
    0: "low",
    1: "medium",
    2: "high",
}

# These MUST match prepare_ml_features.py
FEATURE_COLUMNS = [
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
]


# ============================================================
# MODEL LOADING
# ============================================================

def find_model(model_name: str) -> Path:
    """
    Locate a trained model under models/.
    Supports common joblib/pkl naming variants.
    """

    candidates = [
        MODEL_DIR / f"{model_name}.joblib",
        MODEL_DIR / f"{model_name}.pkl",
        MODEL_DIR / f"{model_name}_model.joblib",
        MODEL_DIR / f"{model_name}_model.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Recursive fallback
    matches = list(MODEL_DIR.rglob(f"{model_name}.joblib"))
    matches += list(MODEL_DIR.rglob(f"{model_name}.pkl"))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find trained model for '{model_name}' under:\n"
        f"{MODEL_DIR}"
    )


def load_model(model_name: str):
    path = find_model(model_name)

    print(f"Loading model:")
    print(f"  {path}")

    model = joblib.load(path)

    return model


# ============================================================
# DATA LOADING
# ============================================================

def load_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_FILE}"
        )

    print(f"Loading feature dataset:")
    print(f"  {FEATURE_FILE}")

    df = pd.read_csv(FEATURE_FILE)

    required = {
        "person_id",
        "job_id",
        "combined_score",
        "skill_match_ratio",
        "sbert_similarity",
        "matching_skills",
        "missing_skills",
        *FEATURE_COLUMNS,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Feature dataset is missing required columns:\n"
            + ", ".join(missing)
        )

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()

    # Match the training preparation behavior.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


# ============================================================
# MODEL SCORE
# ============================================================

def get_model_score(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
        relevance_score
        predicted_class
    """

    # Preferred route for the current 3-class models.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        probabilities = np.asarray(probabilities)

        if probabilities.ndim != 2 or probabilities.shape[1] < 3:
            raise ValueError(
                "Expected a 3-class classifier with probabilities for "
                "low / medium / high."
            )

        # Continuous relevance:
        #
        # low    = 0.0
        # medium = 0.5
        # high   = 1.0
        #
        # This gives a smooth score rather than using only argmax.
        relevance = (
            probabilities[:, 0] * 0.0
            + probabilities[:, 1] * 0.5
            + probabilities[:, 2] * 1.0
        )

        predicted_class = np.argmax(probabilities, axis=1)

        return relevance, predicted_class

    # LinearSVC does not expose predict_proba.
    # Use decision_function and convert the multiclass decision
    # scores into a normalized continuous relevance score.
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X))

        if decision.ndim == 1:
            # Binary fallback.
            scaled = 1.0 / (1.0 + np.exp(-decision))
            predicted = (scaled >= 0.5).astype(int)

            return scaled, predicted

        if decision.shape[1] >= 3:
            # Softmax normalization.
            decision = decision - decision.max(axis=1, keepdims=True)

            exp_scores = np.exp(decision)
            probabilities = exp_scores / exp_scores.sum(
                axis=1,
                keepdims=True,
            )

            relevance = (
                probabilities[:, 0] * 0.0
                + probabilities[:, 1] * 0.5
                + probabilities[:, 2] * 1.0
            )

            predicted_class = np.argmax(probabilities, axis=1)

            return relevance, predicted_class

    raise TypeError(
        "Model does not provide predict_proba() or decision_function()."
    )


# ============================================================
# EXPLANATION HELPERS
# ============================================================

def parse_skill_list(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # The feature-generation pipeline normally stores comma-separated
    # skill strings. Also tolerate semicolon-separated values.
    if ";" in text:
        parts = text.split(";")
    else:
        parts = text.split(",")

    return [
        item.strip()
        for item in parts
        if item.strip()
    ]


def normalize_percent(value: float) -> float:
    return round(float(value) * 100.0, 2)


# ============================================================
# RANKING
# ============================================================

def rank_candidates(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 10,
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    model = load_model(model_name)

    X = prepare_features(df)

    model_score, predicted_class = get_model_score(model, X)

    result = df[
        [
            "person_id",
            "job_id",
            "title",
            "category",
            "experience_years",
            "num_skills",
            "num_matching_skills",
            "num_required_skills",
            "skill_match_ratio",
            "sbert_similarity",
            "combined_score",
            "matching_skills",
            "missing_skills",
        ]
    ].copy()

    result["ml_score"] = model_score
    result["predicted_class"] = predicted_class
    result["predicted_relevance"] = [
        LABEL_NAMES.get(int(x), str(x))
        for x in predicted_class
    ]

    # --------------------------------------------------------
    # Production ranking score
    #
    # ML score is the primary learned signal.
    # Existing matching signals remain visible and contribute
    # to the final ranking.
    # --------------------------------------------------------

    result["final_score"] = (
        0.60 * result["ml_score"]
        + 0.25 * result["skill_match_ratio"]
        + 0.15 * result["sbert_similarity"]
    )

    result = result.sort_values(
        by=[
            "final_score",
            "ml_score",
            "skill_match_ratio",
            "sbert_similarity",
        ],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    result["rank"] = np.arange(1, len(result) + 1)

    # --------------------------------------------------------
    # Human-friendly percentage fields
    # --------------------------------------------------------

    result["ml_score_pct"] = result["ml_score"].apply(normalize_percent)
    result["skill_match_pct"] = result["skill_match_ratio"].apply(
        normalize_percent
    )
    result["sbert_similarity_pct"] = result["sbert_similarity"].apply(
        normalize_percent
    )
    result["combined_score_pct"] = result["combined_score"].apply(
        normalize_percent
    )
    result["final_score_pct"] = result["final_score"].apply(
        normalize_percent
    )

    # --------------------------------------------------------
    # Keep the requested Top-K
    # --------------------------------------------------------

    result = result.head(top_k).copy()

    return result


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: pd.DataFrame,
    model_name: str,
    job_id: str,
) -> None:

    print()
    print("=" * 90)
    print("RESUME SCREENING — CANDIDATE RANKING")
    print("=" * 90)

    print(f"Job ID       : {job_id}")
    print(f"Model        : {model_name}")
    print(f"Candidates   : {len(results)}")

    if results.empty:
        print()
        print("No candidates found.")
        return

    print()
    print("-" * 90)
    print("RANKED CANDIDATES")
    print("-" * 90)

    for _, row in results.iterrows():

        print()
        print(
            f"#{int(row['rank'])}  "
            f"Candidate {int(row['person_id'])}"
        )

        print(
            f"    Final Score       : {row['final_score_pct']:.2f}%"
        )

        print(
            f"    ML Score          : {row['ml_score_pct']:.2f}%"
        )

        print(
            f"    Relevance         : "
            f"{row['predicted_relevance']}"
        )

        print(
            f"    Skill Match       : "
            f"{row['skill_match_pct']:.2f}%"
        )

        print(
            f"    SBERT Similarity  : "
            f"{row['sbert_similarity_pct']:.2f}%"
        )

        print(
            f"    Combined Score    : "
            f"{row['combined_score_pct']:.2f}%"
        )

        print(
            f"    Experience        : "
            f"{row['experience_years']:.1f} years"
        )

        print(
            f"    Matching Skills   : "
            f"{int(row['num_matching_skills'])}/"
            f"{int(row['num_required_skills'])}"
        )

        matched = parse_skill_list(row["matching_skills"])
        missing = parse_skill_list(row["missing_skills"])

        print(
            "    Matched Skills    : "
            + (", ".join(matched) if matched else "None")
        )

        print(
            "    Missing Skills    : "
            + (", ".join(missing) if missing else "None")
        )

    print()
    print("=" * 90)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: pd.DataFrame,
    job_id: str,
    model_name: str,
) -> Path:

    output_dir = MODEL_DIR / "ranking_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"{job_id}_{model_name}_ranking.csv"
    )

    results.to_csv(output_file, index=False)

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Rank candidates for a job."
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="Job ID, e.g. JOB01090",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidates to return.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "logistic_regression",
            "linear_svm",
            "random_forest",
            "lightgbm",
            "xgboost",
        ],
        help="Model used for ranking.",
    )

    args = parser.parse_args()

    print("=" * 90)
    print("RESUME SCREENING — PRODUCTION RANKING ENGINE")
    print("=" * 90)

    print()
    print(f"Requested job : {args.job_id}")
    print(f"Model         : {args.model}")
    print(f"Top-K         : {args.top_k}")

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    df = load_features()

    job_df = df[
        df["job_id"].astype(str) == str(args.job_id)
    ].copy()

    if job_df.empty:
        available = (
            df["job_id"]
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"Job '{args.job_id}' was not found.\n\n"
            f"Example available job IDs:\n"
            + "\n".join(f"  {x}" for x in available)
        )

    print()
    print(f"Candidates found for job: {len(job_df)}")

    results = rank_candidates(
        job_df,
        model_name=args.model,
        top_k=args.top_k,
    )

    print_results(
        results,
        model_name=args.model,
        job_id=args.job_id,
    )

    output_file = save_results(
        results,
        job_id=args.job_id,
        model_name=args.model,
    )

    print()
    print("Saved ranking:")
    print(f"  {output_file}")

    print()
    print("RANKING COMPLETE")


if __name__ == "__main__":
    main()