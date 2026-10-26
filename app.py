
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

MODEL_PATH = Path("artifacts/chatgpt_satisfaction_pipeline.joblib")

FEATURES = [
    "Age",
    "Country",
    "Plan",
    "Primary_Use_Case",
    "Device",
    "Sessions",
    "Avg_Session_Minutes",
    "Messages_Sent",
    "Files_Uploaded",
    "Images_Generated",
    "Favorite_Model",
    "Monthly_Spend_USD",
    "Notifications_Enabled",
    "Dark_Mode"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained ChatGPT satisfaction model was not found."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "chatgpt-satisfaction-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction = str(model.predict(sample)[0])

        return jsonify({
            "prediction": prediction,
            "target": "Satisfaction"
        }), 200

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": "Invalid input values",
            "details": str(error)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
