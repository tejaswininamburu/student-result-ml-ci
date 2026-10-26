
import json
import os
import sys

MODEL_FILE = "artifacts/chatgpt_satisfaction_pipeline.joblib"
METRICS_FILE = "outputs/metrics.json"
MIN_ACCURACY = 0.10


def main():
    if not os.path.exists(MODEL_FILE):
        print("Quality Gate FAILED: Model file is missing")
        sys.exit(1)

    if not os.path.exists(METRICS_FILE):
        print("Quality Gate FAILED: Metrics file is missing")
        sys.exit(1)

    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

    accuracy = metrics.get("accuracy")

    if accuracy is None:
        print("Quality Gate FAILED: Accuracy is missing")
        sys.exit(1)

    print("Model accuracy:", accuracy)
    print("Minimum required accuracy:", MIN_ACCURACY)

    if accuracy < MIN_ACCURACY:
        print("Quality Gate FAILED")
        sys.exit(1)

    print("Quality Gate PASSED")


if __name__ == "__main__":
    main()
