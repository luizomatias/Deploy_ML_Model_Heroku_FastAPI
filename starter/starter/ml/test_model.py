import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from starter.ml.data import process_data
from starter.ml.model import train_model


@pytest.fixture
def data():
    "input data to test"
    df = pd.read_csv("starter/data/census_clean.csv")

    return df


@pytest.fixture
def data_split(data):
    "Split data to test"
    train, test = train_test_split(data, test_size=0.20, random_state=0)

    return train, test


def test_data(data):
    "test data file"
    dataframe = data

    assert len(dataframe.columns) == 15
    assert all(dataframe.columns == [x.strip(" ") for x in dataframe.columns])


def test_data_shape(data):
    """If your data is assumed to have no null values then this is a valid test."""
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_slice_averages_std(data):
    """Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
    for class_name in data["education"].unique():
        avg_value = data[data["education"] == class_name]["age"].mean()
        std_value = data[data["education"] == class_name]["age"].std()
        assert (
            50 > avg_value > 30
        ), f"For {class_name}, average of {avg_value} not between 30 and 50."

        assert (
            17 > std_value > 10
        ), f"For {class_name}, std of {std_value} not between 10 and 17."


def test_train_model(data_split):
    "Test train_model"

    train, test = data_split

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.
    train_model(X_train, y_train)

    assert os.path.exists("starter/model/model_v1.pkl")
