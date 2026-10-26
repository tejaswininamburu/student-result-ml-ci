from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# ⚠️ REPLACE THIS with your actual model filename
MODEL_PATH = Path("chatgpt_model.pkl")

# ⚠️ REPLACE THIS with your exact feature column names
FEATURES = [
    "age", 
    "usage_hours", 
    "subscription_type"
]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} was not found. Run training first.")
    return joblib.load(MODEL_PATH)

@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "chatgpt-users-prediction"
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [feature for feature in FEATURES if feature not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    model = load_model()
    
    prediction_code = int(model.predict(sample)[0])
    
    # ⚠️ Adjust based on what your model predicts (e.g. 1 vs 0)
    prediction = "HIGH_ENGAGEMENT" if prediction_code == 1 else "LOW_ENGAGEMENT"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
