
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "chatgpt-users.csv"
MODEL_FILE = "artifacts/chatgpt_satisfaction_pipeline.joblib"
METRICS_FILE = "outputs/metrics.json"
DATA_OUTPUT = "outputs/chatgpt-users.csv"
TARGET = "Satisfaction"


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing")

    data = data.dropna(subset=[TARGET]).copy()

    if data[TARGET].nunique() < 2:
        raise ValueError("At least two Satisfaction classes are required")

    os.makedirs("outputs", exist_ok=True)
    data.to_csv(DATA_OUTPUT, index=False)

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
    y = data[TARGET].astype(str)

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    class_counts = y.value_counts()
    stratify_values = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_values
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "total_records": int(len(data)),
        "target_column": TARGET,
        "classes": sorted(y.unique().tolist()),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )
    }

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model training completed")
    print("Accuracy:", accuracy)
    print("Model saved:", MODEL_FILE)
    print("Metrics saved:", METRICS_FILE)
    print("Dataset copy saved:", DATA_OUTPUT)


if __name__ == "__main__":
    main()
