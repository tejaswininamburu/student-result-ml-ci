
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


DATA_PATH = Path("chatgpt-users.csv")
ARTIFACT_DIR = Path("artifacts")
TARGET = "Satisfaction"

DROP_COLUMNS = [
    "User_ID",
    "Name",
    "Signup_Date",
    "Last_Active_Date",
]


def build_pipeline(data):
    features = data.drop(
        columns=[TARGET] + DROP_COLUMNS,
        errors="ignore",
    )

    numeric_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = features.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    return model, features


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing"
        )

    data = data.dropna(subset=[TARGET])

    if data[TARGET].nunique() < 2:
        raise ValueError(
            "At least two satisfaction classes are required"
        )

    model, features = build_pipeline(data)
    target = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "target": TARGET,
        "classes": sorted(target.astype(str).unique().tolist()),
    }

    with open(ARTIFACT_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Training completed.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    train_model()
