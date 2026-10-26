
import pandas as pd

from train_model import DATA_PATH, TARGET, build_pipeline


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_has_satisfaction_target():
    data = pd.read_csv(DATA_PATH)

    assert TARGET in data.columns
    assert data[TARGET].notna().all()
    assert data[TARGET].nunique() >= 2


def test_pipeline_trains_and_predicts():
    data = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])

    model, features = build_pipeline(data)

    X = features.iloc[:40]
    y = data[TARGET].iloc[:40]

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset(set(data[TARGET]))


def test_dataset_has_expected_activity_columns():
    data = pd.read_csv(DATA_PATH)

    expected = [
        "Sessions",
        "Messages_Sent",
        "Files_Uploaded",
        "Satisfaction",
    ]

    for column in expected:
        assert column in data.columns
