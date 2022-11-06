from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_method():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "greeting"}


def test_lower_prediction():
    response = client.post(
        "/inference_main",
        json={
            "age": 32,
            "workclass": "Private",
            "fnlgt": 205019,
            "education": "Assoc-acdm",
            "education-num": 12,
            "marital-status": "Never-married",
            "occupation": "Sales",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States"
        },
    )
    assert response.status_code == 200
    assert response.json() == {"salary": "<=50k"}


def test_higher_prediction():
    response = client.post(
        "/inference_main",
        json={
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States"
        },
    )
    assert response.status_code == 200
    assert response.json() == {"salary": ">50k"}
