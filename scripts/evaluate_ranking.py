from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = ROOT / "data" / "evaluation"
ML_DIR = ROOT / "data" / "ml"
MODEL_DIR = ROOT / "models"
OUT_DIR = MODEL_DIR / "ranking_evaluation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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

MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "lightgbm",
    "xgboost",
]

SPLITS = [
    "random",
    "candidate_holdout",
    "job_holdout",
    "candidate_job_holdout",
]


def dcg(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(values) + 2)
    )

    return float(
        np.sum(
            (2 ** values - 1) / discounts
        )
    )


def ndcg_at_k(values, k):
    values = np.asarray(values)

    if len(values) == 0:
        return 0.0

    actual = dcg(values[:k])

    ideal = dcg(
        np.sort(values)[::-1][:k]
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def precision_at_k(values, k):
    values = np.asarray(values)[:k]

    if len(values) == 0:
        return 0.0

    return float(np.mean(values == 2))


def hit_rate_at_k(values, k):
    values = np.asarray(values)[:k]

    if len(values) == 0:
        return 0.0

    return float(np.any(values == 2))


def reciprocal_rank(values):
    for rank, value in enumerate(values, start=1):
        if value == 2:
            return 1.0 / rank

    return 0.0


def average_precision(values):
    values = np.asarray(values)

    total_relevant = np.sum(values == 2)

    if total_relevant == 0:
        return 0.0

    hits = 0
    score = 0.0

    for rank, value in enumerate(values, start=1):

        if value == 2:
            hits += 1
            score += hits / rank

    return score / total_relevant


def load_data(split):

    evaluation_file = (
        EVAL_DIR / f"{split}_test.csv"
    )

    matching_file = (
        ML_DIR / f"{split}_test_matching.csv"
    )

    print(f"\nEvaluation file:")
    print(evaluation_file)

    print("Matching file:")
    print(matching_file)

    if not evaluation_file.exists():
        raise FileNotFoundError(
            f"Missing: {evaluation_file}"
        )

    if not matching_file.exists():
        raise FileNotFoundError(
            f"Missing: {matching_file}"
        )

    evaluation = pd.read_csv(
        evaluation_file
    )

    matching = pd.read_csv(
        matching_file
    )

    print(
        f"Evaluation rows: {len(evaluation):,}"
    )

    print(
        f"Matching rows:   {len(matching):,}"
    )

    required_eval = [
        "person_id",
        "job_id",
        "combined_score",
        "sbert_similarity",
        "skill_match_ratio",
    ]

    required_matching = FEATURES + ["label"]

    missing_eval = [
        c for c in required_eval
        if c not in evaluation.columns
    ]

    missing_matching = [
        c for c in required_matching
        if c not in matching.columns
    ]

    if missing_eval:
        raise ValueError(
            f"{split}: evaluation file missing: "
            + ", ".join(missing_eval)
        )

    if missing_matching:
        raise ValueError(
            f"{split}: ML file missing: "
            + ", ".join(missing_matching)
        )

    if len(evaluation) != len(matching):
        raise ValueError(
            f"{split}: row counts do not match"
        )

    labels_from_band = (
        evaluation["pair_band"]
        .map(
            {
                "low": 0,
                "medium": 1,
                "high": 2,
            }
        )
    )

    if labels_from_band.isna().any():
        raise ValueError(
            f"{split}: invalid pair_band values"
        )

    mismatch = (
        labels_from_band.astype(int)
        != matching["label"].astype(int)
    ).sum()

    if mismatch:
        raise ValueError(
            f"{split}: {mismatch} label mismatches"
        )

    data = pd.DataFrame()

    data["person_id"] = evaluation["person_id"]
    data["job_id"] = evaluation["job_id"]

    data["combined_score"] = (
        evaluation["combined_score"]
    )

    data["sbert_similarity"] = (
        evaluation["sbert_similarity"]
    )

    data["skill_match_ratio"] = (
        evaluation["skill_match_ratio"]
    )

    for column in FEATURES:
        data[column] = matching[column].values

    data["label"] = matching["label"].values

    return data


def get_model_score(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if probabilities.ndim == 2:

            if probabilities.shape[1] >= 3:
                return probabilities[:, 2]

            return probabilities[:, -1]

        return probabilities

    if hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        if np.ndim(scores) == 2:

            if scores.shape[1] >= 3:
                return scores[:, 2]

            return scores[:, -1]

        return scores

    return model.predict(X)


def evaluate_method(
    data,
    split,
    method,
    scores
):

    frame = data[
        [
            "job_id",
            "person_id",
            "label"
        ]
    ].copy()

    frame["score"] = np.asarray(
        scores,
        dtype=float
    )

    frame = frame.sort_values(
        [
            "job_id",
            "score",
            "person_id"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )

    results = []

    for job_id, group in frame.groupby(
        "job_id",
        sort=False
    ):

        relevance = (
            group["label"]
            .astype(int)
            .to_numpy()
        )

        row = {
            "split": split,
            "method": method,
            "job_id": job_id,
            "num_candidates": len(group),
            "ndcg@5": ndcg_at_k(
                relevance, 5
            ),
            "ndcg@10": ndcg_at_k(
                relevance, 10
            ),
            "precision@5": precision_at_k(
                relevance, 5
            ),
            "precision@10": precision_at_k(
                relevance, 10
            ),
            "hit_rate@5": hit_rate_at_k(
                relevance, 5
            ),
            "hit_rate@10": hit_rate_at_k(
                relevance, 10
            ),
            "mrr": reciprocal_rank(
                relevance
            ),
            "average_precision":
                average_precision(
                    relevance
                )
        }

        results.append(row)

    return pd.DataFrame(results)


print("=" * 80)
print("RESUME SCREENING — RANKING EVALUATION")
print("=" * 80)

print("\nCORRECTED VERSION: 2026-08-26")
print("Using evaluation CSV + matching CSV.")
print("NOT using metadata CSV.")

all_results = []

for split in SPLITS:

    print("\n" + "=" * 80)
    print(f"SPLIT: {split.upper()}")
    print("=" * 80)

    data = load_data(split)

    print(
        f"Jobs: {data['job_id'].nunique():,}"
    )

    print(
        f"People: {data['person_id'].nunique():,}"
    )

    print("\nLabel distribution:")

    print(
        data["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    baselines = {
        "combined_score":
            data["combined_score"].values,

        "sbert_similarity":
            data["sbert_similarity"].values,

        "skill_match_ratio":
            data["skill_match_ratio"].values,
    }

    for method, scores in baselines.items():

        print(
            f"\n  Evaluating {method}..."
        )

        result = evaluate_method(
            data,
            split,
            method,
            scores
        )

        all_results.append(result)

    X = data[FEATURES]

    for model_name in MODELS:

        model_file = (
            MODEL_DIR
            / f"{model_name}.joblib"
        )

        if not model_file.exists():

            print(
                f"\n  WARNING: missing "
                f"{model_file.name}"
            )

            continue

        print(
            f"\n  Evaluating {model_name}..."
        )

        model = joblib.load(
            model_file
        )

        scores = get_model_score(
            model,
            X
        )

        result = evaluate_method(
            data,
            split,
            model_name,
            scores
        )

        all_results.append(result)


per_job = pd.concat(
    all_results,
    ignore_index=True
)

per_job_file = (
    OUT_DIR
    / "per_job_ranking_metrics.csv"
)

per_job.to_csv(
    per_job_file,
    index=False
)


metrics = [
    "ndcg@5",
    "ndcg@10",
    "precision@5",
    "precision@10",
    "hit_rate@5",
    "hit_rate@10",
    "mrr",
    "average_precision"
]

summary = (
    per_job
    .groupby(
        ["split", "method"]
    )[metrics]
    .mean()
    .reset_index()
)

summary_file = (
    OUT_DIR
    / "ranking_metrics_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


strict = summary[
    summary["split"]
    == "candidate_job_holdout"
].copy()

strict = strict.sort_values(
    [
        "ndcg@5",
        "ndcg@10",
        "mrr"
    ],
    ascending=False
)

strict_file = (
    OUT_DIR
    / "strict_ranking_comparison.csv"
)

strict.to_csv(
    strict_file,
    index=False
)


if len(strict):

    best = strict.iloc[0]

    best_info = {
        "best_method":
            best["method"],
        "selection_split":
            "candidate_job_holdout",
        "selection_metric":
            "ndcg@5"
    }

    for metric in metrics:
        best_info[metric] = float(
            best[metric]
        )

    with open(
        OUT_DIR / "best_ranking_method.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_info,
            f,
            indent=2
        )


print("\n" + "=" * 80)
print("RANKING EVALUATION COMPLETE")
print("=" * 80)

print("\nSummary:")

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)

print("\n" + "=" * 80)
print("STRICT CANDIDATE + JOB HOLDOUT")
print("=" * 80)

print(
    strict.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)

if len(strict):

    print("\nBEST STRICT RANKING METHOD")

    print(
        f"Method: {best['method']}"
    )

    print(
        f"NDCG@5: {best['ndcg@5']:.4f}"
    )

    print(
        f"NDCG@10: {best['ndcg@10']:.4f}"
    )

    print(
        f"Precision@5: "
        f"{best['precision@5']:.4f}"
    )

    print(
        f"MRR: {best['mrr']:.4f}"
    )

print("\nSaved files:")

print(per_job_file)
print(summary_file)
print(strict_file)
print(
    OUT_DIR
    / "best_ranking_method.json"
)

print("\nDONE.")