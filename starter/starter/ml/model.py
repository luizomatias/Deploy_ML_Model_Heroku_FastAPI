from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import List, Dict
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "starter/model/model_v1.pkl")
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def performance_model_slice_data(
    model,
    data: pd.DataFrame,
    categorical_features_list: List[str],
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
):
    """Performance model slice in data.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    data : pd.DataFrame
        Data to be used.
    categorical_features_list: List[str]
        Name of the categorical features
    encoder: OneHotEncoder
        Trained OneHotEncoder
    lb: LabelBinarizer
        Trained LabelBinarizer
    Returns
    -------
    preds : Dict
        A dictionary containing model predictions for the
        categorical features
    """

    preds = {}

    for column in categorical_features_list:
        for feature in data[column].unique():
            data_temp = data[data[column] == feature]
            X_feature, y_feature, encoder, lb = process_data(
                data_temp,
                categorical_features=categorical_features_list,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            predict = inference(model, X_feature)
            precision, recall, fbeta = compute_model_metrics(y_feature, predict)

            preds[feature] = {"precision": precision, "recall": recall, "fbeta": fbeta}

    return preds
