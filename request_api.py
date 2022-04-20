import requests
import json
from main import Features
import pandas as pd

data = Features().dict(by_alias=True)
response = requests.post("https://ml-deploying-fastapi.herokuapp.com/predictions", json = data)

print(f'Status Post Request: {response.status_code}')
print()
print(f'Prediction: {response.json()}')
