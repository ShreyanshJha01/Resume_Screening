import json
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ml"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "random_train_matching.csv"
TEST_FILE = DATA_DIR / "random_test_matching.csv"

RANDOM_STATE = 42
EXPERIMENT_NAME = "Resume Screening - ML Classification"
MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow_v2.db'}"

# ============================================================
# FEATURES
# ============================================================

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

TARGET = "label"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("RESUME SCREENING — ML MODEL TRAINING")
print("=" * 80)

print("\nLoading training data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")

# ============================================================
# VALIDATE FEATURES AND TARGET
# ============================================================

missing_train_features = [
    feature for feature in FEATURES if feature not in train.columns
]

missing_test_features = [
    feature for feature in FEATURES if feature not in test.columns
]

if missing_train_features:
    raise ValueError(
        "Missing features in training data: "
        + ", ".join(missing_train_features)
    )

if missing_test_features:
    raise ValueError(
        "Missing features in test data: "
        + ", ".join(missing_test_features)
    )

if TARGET not in train.columns:
    raise ValueError(
        "Missing target column 'label' in training data. "
        "Run prepare_ml_features.py first."
    )

if TARGET not in test.columns:
    raise ValueError(
        "Missing target column 'label' in test data. "
        "Run prepare_ml_features.py first."
    )

# ============================================================
# X / y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

print(f"\nNumber of features: {len(FEATURES)}")
print(f"Classes: {sorted(y_train.unique())}")

# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ]),

    "linear_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    ),

    "lightgbm": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        class_weight="balanced",
    ),

    "xgboost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    ),
}

# ============================================================
# MLFLOW SETUP
# ============================================================

print("\n" + "=" * 80)
print("CONFIGURING MLFLOW")
print("=" * 80)

# Store MLflow artifacts inside this project
mlruns_dir = PROJECT_ROOT / "mlruns"
mlruns_dir.mkdir(parents=True, exist_ok=True)

# SQLite tracking database
mlflow.set_tracking_uri(MLFLOW_URI)

# Create experiment with an explicit artifact location
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=mlruns_dir.as_uri()
    )

mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow tracking URI:\n{MLFLOW_URI}")
print(f"\nExperiment:\n{EXPERIMENT_NAME}")
print(f"\nArtifact directory:\n{mlruns_dir}")

# ============================================================
# TRAIN MODELS
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 80)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 80)

    with mlflow.start_run(run_name=model_name) as run:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision_macro = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        recall_macro = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        f1_macro = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        f1_weighted = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        print(f"\nAccuracy       : {accuracy:.4f}")
        print(f"Macro Precision: {precision_macro:.4f}")
        print(f"Macro Recall   : {recall_macro:.4f}")
        print(f"Macro F1       : {f1_macro:.4f}")
        print(f"Weighted F1    : {f1_weighted:.4f}")

        print("\nClassification Report:")

        print(classification_report(
            y_test,
            y_pred,
            target_names=["low", "medium", "high"],
            zero_division=0,
        ))

        print("Confusion Matrix:")

        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)

        # ----------------------------------------------------
        # MLflow parameters
        # ----------------------------------------------------

        mlflow.log_param("feature_set", "matching_12")
        mlflow.log_param("num_features", len(FEATURES))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("target", TARGET)
        mlflow.log_param(
            "label_mapping",
            "low=0, medium=1, high=2",
        )

        # ----------------------------------------------------
        # MLflow metrics
        # ----------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # ----------------------------------------------------
        # Classification report artifact
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            target_names=["low", "medium", "high"],
            output_dict=True,
            zero_division=0,
        )

        report_file = (
            MODEL_DIR / f"{model_name}_classification_report.json"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact(str(report_file))

        # ----------------------------------------------------
        # Confusion matrix artifact
        # ----------------------------------------------------

        matrix_file = (
            MODEL_DIR / f"{model_name}_confusion_matrix.csv"
        )

        pd.DataFrame(
            matrix,
            index=["actual_low", "actual_medium", "actual_high"],
            columns=["pred_low", "pred_medium", "pred_high"],
        ).to_csv(matrix_file)

        mlflow.log_artifact(str(matrix_file))

        # ----------------------------------------------------
        # Local model artifact
        # ----------------------------------------------------

        model_file = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_file)

        # ----------------------------------------------------
        # MLflow model artifact
        # ----------------------------------------------------

        if model_name in [
            "logistic_regression",
            "linear_svm",
            "random_forest",
        ]:
            mlflow.sklearn.log_model(model, name="model")

        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(model, name="model")

        elif model_name == "xgboost":
            mlflow.xgboost.log_model(model, name="model")

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "run_id": run.info.run_id,
        })

        print(f"\nMLflow Run ID:\n{run.info.run_id}")

# ============================================================
# SAVE COMPARISON
# ============================================================

results_df = pd.DataFrame(results).sort_values(
    "f1_macro",
    ascending=False,
)

results_file = MODEL_DIR / "model_comparison_random_test.csv"

results_df.to_csv(results_file, index=False)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE")
print("=" * 80)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print(f"\nModels saved to:\n{MODEL_DIR}")
print(f"\nMLflow database:\n{PROJECT_ROOT / 'mlflow.db'}")

print("\n" + "=" * 80)
