import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from starter.starter.ml.data import process_data
import pandas as pd



if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Import model, encoder and lb
model = joblib.load('starter/model/model_v1.pkl')
encoder = joblib.load('starter/model/encoder_v1.pkl')
lb = joblib.load('starter/model/lb_v1.pkl')

#categorical features

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Features to ingest the body from POST
class Features(BaseModel):
    age: int = 28
    workclass: str = 'Private'
    fnlgt: int = 338409
    education: str = 'Bachelors'
    education_num: int = 13
    marital_status: str = 'Married-civ-spouse'
    occupation: str = 'Prof-specialty'
    relationship: str = 'Wife'
    race: str = 'Black'
    sex: str = 'Female'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int  = 40
    native_country: str = 'Cuba'


app = FastAPI()


@app.get("/")
async def welcome_message():
    return {"message": "Welcome to classification model on publicly available Census Bureau data! :)"}


@app.post("/predictions")
async def predict_model(features: Features):

    data = pd.DataFrame(data = features.dict(by_alias=True), index=[0])

    X, _, _, _ = process_data(
                                data,
                                categorical_features=cat_features,
                                label=None,
                                training=False,
                                encoder=encoder,
                                lb=lb)

    predict = model.predict(X)

    if predict[0] == 1:
        predict = "Salary > 50k"
    else:
        predict = "Salary <= 50k"

    return {'predict': predict}