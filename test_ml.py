import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def processed_data():
    """Load a sample of the census data and process it for training."""
    data = pd.read_csv("data/census.csv").sample(n=2000, random_state=42)
    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    return X, y


@pytest.fixture(scope="module")
def trained_model(processed_data):
    """Train a model once and share it across tests."""
    X, y = processed_data
    return train_model(X, y)


def test_train_model_returns_random_forest(trained_model):
    """
    train_model should return a fitted RandomForestClassifier.
    """
    assert isinstance(trained_model, RandomForestClassifier)
    # a fitted sklearn estimator exposes attributes ending in "_"
    assert hasattr(trained_model, "classes_")


def test_inference_output_shape_and_values(trained_model, processed_data):
    """
    inference should return a numpy array with one binary prediction per row.
    """
    X, y = processed_data
    preds = inference(trained_model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    """
    compute_model_metrics should return the expected precision, recall,
    and F1 for a hand-checkable set of labels and predictions.
    """
    y = np.array([1, 1, 1, 0, 0, 0])
    preds = np.array([1, 1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # TP=2, FP=1, FN=1 -> precision = 2/3, recall = 2/3, F1 = 2/3
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)
