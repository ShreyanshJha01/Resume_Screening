from pathlib import Path

import pandas as pd
import joblib
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "resume_jd_matching_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_best.pkl"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# =========================================================
# FEATURES
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

TARGET_COLUMN = "relevance_label"


# =========================================================
# LOAD DATA
# =========================================================

print("Loading matching dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# CREATE X AND Y
# =========================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

X = X.fillna(0)


print("\nFeatures used for final LightGBM model:")

for feature in FEATURE_COLUMNS:
    print("-", feature)

print("\nTarget:", TARGET_COLUMN)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# =========================================================
# FINAL LIGHTGBM MODEL
# =========================================================

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=-1
)


# =========================================================
# MLflow SETUP
# =========================================================

tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Resume_Job_Matching")


# =========================================================
# TRAIN FINAL MODEL
# =========================================================

print("\nTraining final LightGBM model...")

with mlflow.start_run(run_name="LightGBM_Final"):

    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # Test prediction
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # -----------------------------------------------------
    # Print metrics
    # -----------------------------------------------------

    print("\nFinal LightGBM Performance")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # -----------------------------------------------------
    # MLflow parameters
    # -----------------------------------------------------

    mlflow.log_param("model_name", "LightGBM")
    mlflow.log_param("num_features", len(FEATURE_COLUMNS))
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.03)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("num_leaves", 31)
    mlflow.log_param("min_child_samples", 20)
    mlflow.log_param("subsample", 0.9)
    mlflow.log_param("colsample_bytree", 0.9)

    # -----------------------------------------------------
    # MLflow metrics
    # -----------------------------------------------------

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # -----------------------------------------------------
    # Save model locally
    # -----------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    # -----------------------------------------------------
    # Log LightGBM model to MLflow
    # -----------------------------------------------------

    mlflow.lightgbm.log_model(
        model,
        name="lightgbm_final"
    )


# =========================================================
# COMPLETION
# =========================================================

print("\n" + "=" * 60)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Model:", "LightGBM")
print(f"Accuracy: {accuracy:.4f}")
print("Model path:", MODEL_PATH)
print("MLflow database:", MLFLOW_DB)