# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_model_slice_data,
)
from ml.data import process_data
import pandas as pd
import joblib
import logging
import numpy as np
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info("Loading data...")
data = pd.read_csv("starter/data/census_clean.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Split data...")
train, test = train_test_split(data, test_size=0.20)

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

logger.info("Processing train data...")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logger.info("Processing test data...")
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
logger.info("Saving model...")
model = train_model(X_train, y_train)
joblib.dump(encoder, "starter/model/encoder_v1.pkl")
joblib.dump(lb, "starter/model/lb_v1.pkl")

# Predict test data
logger.info("Predicting on test data...")
test_predict = inference(model, X_test)

# Metrics
logger.info("Calculating metrics...")
precision, recall, fbeta = compute_model_metrics(y_test, test_predict)

logger.info(
    f"Precision: {np.round(precision, 2) * 100}%, \
Recall: {np.round(recall, 2) * 100}%, \
Fbeta: {np.round(fbeta, 2) * 100}%"
)

metrics_dict = {"Precision": precision, "Recall": recall, "Fbeta": fbeta}

with open("starter/data/metrics_dict.json", "w") as fp:
    json.dump(metrics_dict, fp)

# Metrics data slice
logger.info("Calculating metrics data slice...")
metrics_data_slice = performance_model_slice_data(
    model, data, cat_features, encoder, lb
)

with open("starter/data/metrics_data_slice.json", "w") as fp:
    json.dump(metrics_data_slice, fp)
