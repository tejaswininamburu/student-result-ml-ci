
import json
from pathlib import Path

import joblib


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def run_quality_gate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model is missing")

    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics file is missing")

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    required_keys = [
        "accuracy",
        "balanced_accuracy",
        "test_rows",
        "train_rows",
        "target",
        "classes",
    ]

    for key in required_keys:
        if key not in metrics:
            raise ValueError(f"Missing metric: {key}")

    for key in ["accuracy", "balanced_accuracy"]:
        value = metrics[key]

        if not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")

        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

    if metrics["target"] != "Satisfaction":
        raise ValueError("Unexpected prediction target")

    if metrics["test_rows"] <= 0:
        raise ValueError("Test dataset is empty")

    if not metrics["classes"]:
        raise ValueError("No satisfaction classes were found")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise ValueError("Loaded model cannot make predictions")

    print("QUALITY GATE PASSED")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(
        "Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    run_quality_gate()
