import requests
import json
from main import Features
import pandas as pd

data = Features().dict(by_alias=True)
response = requests.post("https://ml-deploying-fastapi.herokuapp.com/predictions", json = data)

print(response.json())

