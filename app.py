from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Your model is in the artifacts folder
MODEL_PATH = Path("artifacts/model.joblib")

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

    # Convert incoming JSON to a DataFrame
    sample = pd.DataFrame([data])

    model = load_model()
    prediction = model.predict(sample)[0]
    
    return jsonify({
        "prediction": str(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
