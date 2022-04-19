from fastapi.testclient import TestClient
from main import app, Features
import json
import logging
import requests


client = TestClient(app)


def test_api_get_root():

    response = client.get("/")
    assert response.status_code == 200
    assert response.json()['message'] == "Welcome to classification model on publicly available Census Bureau data! :)"


def test_post_request_target_salary_bigger_50k():

    data = Features().dict(by_alias=True)
    response = client.post("/predictions", json = data)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Salary > 50k'}


def test_post_request_target_salary_smaller_50k_remote():

    data = Features().dict(by_alias=True)
    data['relationship'] = 'Not-in-family'
    data['education'] = 'HS-grad'
    data['workclass'] = 'Self-emp-inc'
    data['hours_per_week'] = 13
    response = requests.post("https://ml-deploying-fastapi.herokuapp.com/predictions", json = data)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Salary <= 50k'}


def test_post_request_target_salary_smaller_50k():

    data = Features().dict(by_alias=True)
    data['relationship'] = 'Not-in-family'
    data['education'] = 'HS-grad'
    data['workclass'] = 'Self-emp-inc'
    data['hours_per_week'] = 13
    response = client.post("/predictions", json = data)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Salary <= 50k'}
