from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier
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

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    # Candidate features
    "resume_word_count",
    "resume_char_count",
    "num_skills",
    "num_experiences",
    "num_education_records",
    "experience_years",

    # Job features
    "num_required_skills",
    "job_word_count",
    "job_char_count",

    # Matching features
    "sbert_similarity"
   
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

# Handle missing values
X = X.fillna(0)

print("\nFeatures used for training:")

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
# DEFINE MODELS
# =========================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                kernel="rbf",
                random_state=42
            )
        )
    ]),

    "LightGBM": LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])
}


# =========================================================
# MLFLOW SETUP
# =========================================================

tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Resume_Job_Matching")

print("\nMLflow tracking URI:")
print(tracking_uri)

print("\nStarting model training...")


# =========================================================
# TRAIN AND EVALUATE
# =========================================================

results = []

best_model_name = None
best_accuracy = 0


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.fit(X_train, y_train)

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        y_pred = model.predict(X_test)

        # -------------------------------------------------
        # CALCULATE METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

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

        # -------------------------------------------------
        # LOG PARAMETERS TO MLFLOW
        # -------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "num_features",
            len(FEATURE_COLUMNS)
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # -------------------------------------------------
        # LOG METRICS TO MLFLOW
        # -------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # -------------------------------------------------
        # LOG MODEL TO MLFLOW
        # -------------------------------------------------

        if model_name == "LightGBM":

            mlflow.lightgbm.log_model(
                model,
                name="model"
            )

        elif model_name == "XGBoost":

            mlflow.xgboost.log_model(
                model,
                name="model"
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model"
            )

        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # -------------------------------------------------
        # PRINT RESULTS
        # -------------------------------------------------

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # -------------------------------------------------
        # FIND BEST MODEL
        # -------------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model_name = model_name


# =========================================================
# MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="accuracy",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# =========================================================
# BEST MODEL
# =========================================================

print("\nBest model:")
print(best_model_name)

print(f"Best accuracy: {best_accuracy:.4f}")


# =========================================================
# MLFLOW LOCATION
# =========================================================

print("\nMLflow tracking database:")
print(MLFLOW_DB)

print("\nTraining completed successfully.")