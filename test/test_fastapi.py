import json
from fastapi.testclient import TestClient
from fastapi import FastAPI

def test_lower50k_prediction():
    app = FastAPI()
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 32,
                "workclass": "Private",
                "fnlwgt": 205019,
                "education": "Assoc-acdm",
                "education_num": 12,
                "marital_status": "Never-married",
                "occupation": "Sales",
                "relationship": "Not-in-family",
                "race": "Black",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 50,
                "native_country": "United-States"
            },
        )
        assert response.status_code == 200, response.json()
        assert response.json() == {"prediction": {"salary": "<=50K"}}