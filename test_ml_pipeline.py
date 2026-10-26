
import json
import os
import unittest
import joblib
import pandas as pd


MODEL_FILE = "artifacts/chatgpt_satisfaction_pipeline.joblib"
METRICS_FILE = "outputs/metrics.json"
DATA_FILE = "chatgpt-users.csv"
TARGET = "Satisfaction"


class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists(MODEL_FILE))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists(METRICS_FILE))

    def test_model_can_be_loaded(self):
        model = joblib.load(MODEL_FILE)
        self.assertTrue(hasattr(model, "predict"))

    def test_model_can_make_predictions(self):
        data = pd.read_csv(DATA_FILE)
        data = data.dropna(subset=[TARGET])

        drop_columns = [
            "User_ID",
            "Name",
            "Signup_Date",
            "Last_Active_Date"
        ]

        X = data.drop(
            columns=[TARGET] + [
                col for col in drop_columns if col in data.columns
            ]
        )

        model = joblib.load(MODEL_FILE)
        predictions = model.predict(X.head(5))

        self.assertEqual(len(predictions), min(5, len(X)))

    def test_metrics_are_valid(self):
        with open(METRICS_FILE, "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreater(metrics["training_records"], 0)
        self.assertGreater(metrics["testing_records"], 0)


if __name__ == "__main__":
    unittest.main()
