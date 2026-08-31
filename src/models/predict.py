from pathlib import Path
import pandas as pd
import joblib


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_best.pkl"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "resume_jd_matching_dataset.csv"
)


# =========================================================
# FEATURES USED BY FINAL MODEL
# =========================================================

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
    "sbert_similarity",
]


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading LightGBM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading matching dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# SELECT FEATURES
# =========================================================

X = df[FEATURE_COLUMNS].copy()

X = X.fillna(0)


# =========================================================
# PREDICT
# =========================================================

predictions = model.predict(X)


# =========================================================
# CONVERT NUMERIC LABEL TO TEXT
# =========================================================

label_mapping = {
    0: "Low Match",
    1: "Medium Match",
    2: "High Match"
}


df["predicted_label"] = predictions

df["predicted_result"] = df["predicted_label"].map(
    label_mapping
)


# =========================================================
# DISPLAY SAMPLE RESULTS
# =========================================================

print("\n" + "=" * 60)
print("RESUME-JOB MATCHING PREDICTIONS")
print("=" * 60)

print(
    df[
        [
            "person_id",
            "job_id",
            "relevance_label",
            "predicted_label",
            "predicted_result"
        ]
    ].head(20).to_string(index=False)
)


# =========================================================
# ACCURACY ON COMPLETE DATA
# =========================================================

correct = (
    df["relevance_label"] == df["predicted_label"]
).mean()

print("\nPrediction accuracy on complete dataset:")
print(f"{correct:.4f}")


print("\nPrediction completed successfully.")