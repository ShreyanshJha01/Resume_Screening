from pathlib import Path
import json
import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"
OUTPUT_DIR = ROOT / "models" / "mlflow_reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
mlflow.set_tracking_uri(TRACKING_URI)

print("=" * 80)
print("RESUME SCREENING — MLFLOW REPORT")
print("=" * 80)

client = mlflow.MlflowClient()

experiments = client.search_experiments()

print("\nMLflow experiments:")
for exp in experiments:
    if exp.name != "Default":
        print(f"  - {exp.name} (ID={exp.experiment_id})")


def collect_runs(experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"\nExperiment not found: {experiment_name}")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )

    records = []

    for run in runs:
        row = {
            "experiment": experiment_name,
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

        for key, value in run.data.params.items():
            row[f"param_{key}"] = value

        for key, value in run.data.metrics.items():
            row[f"metric_{key}"] = value

        records.append(row)

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# 1. TRAINING EXPERIMENT
# ------------------------------------------------------------------

training_experiment = "Resume Screening - ML Classification"

print("\n" + "=" * 80)
print("TRAINING EXPERIMENT")
print("=" * 80)

training_df = collect_runs(training_experiment)

if not training_df.empty:
    print(training_df.to_string(index=False))

    training_path = OUTPUT_DIR / "mlflow_training_runs.csv"
    training_df.to_csv(training_path, index=False)

    print(f"\nSaved:")
    print(f"  {training_path}")


# ------------------------------------------------------------------
# 2. CROSS-SPLIT EVALUATION
# ------------------------------------------------------------------

evaluation_experiment = "Resume Screening - Cross Split Evaluation"

print("\n" + "=" * 80)
print("CROSS-SPLIT EVALUATION EXPERIMENT")
print("=" * 80)

evaluation_df = collect_runs(evaluation_experiment)

if not evaluation_df.empty:
    print(evaluation_df.to_string(index=False))

    evaluation_path = OUTPUT_DIR / "mlflow_cross_split_runs.csv"
    evaluation_df.to_csv(evaluation_path, index=False)

    print(f"\nSaved:")
    print(f"  {evaluation_path}")


# ------------------------------------------------------------------
# 3. MODEL COMPARISON
# ------------------------------------------------------------------

if not training_df.empty:

    model_column = None

    for candidate in [
        "param_model",
        "param_model_name",
        "param_algorithm"
    ]:
        if candidate in training_df.columns:
            model_column = candidate
            break

    metric_columns = [
        "metric_accuracy",
        "metric_precision_macro",
        "metric_recall_macro",
        "metric_f1_macro",
        "metric_f1_weighted"
    ]

    available_metrics = [
        col for col in metric_columns
        if col in training_df.columns
    ]

    if model_column and available_metrics:

        comparison = training_df[
            [model_column] + available_metrics
        ].copy()

        comparison = comparison.rename(
            columns={
                model_column: "model"
            }
        )

        comparison = comparison.sort_values(
            "metric_f1_macro",
            ascending=False
        )

        comparison_path = OUTPUT_DIR / "mlflow_model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        print(comparison.to_string(index=False))

        print(f"\nSaved:")
        print(f"  {comparison_path}")


# ------------------------------------------------------------------
# 4. RUN DETAILS JSON
# ------------------------------------------------------------------

summary = {
    "tracking_uri": TRACKING_URI,
    "training_experiment": training_experiment,
    "evaluation_experiment": evaluation_experiment,
    "training_runs": (
        len(training_df)
        if not training_df.empty
        else 0
    ),
    "evaluation_runs": (
        len(evaluation_df)
        if not evaluation_df.empty
        else 0
    ),
}

summary_path = OUTPUT_DIR / "mlflow_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("MLFLOW REPORT COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(f"  {OUTPUT_DIR}")

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {path.name}")