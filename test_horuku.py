import json
import requests

sample = {
    "age": 53,
    "workclass": "Private",
    "fnlgt": 234721,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Married-civ-spouse",
    "occupation": "Handlers-cleaners",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
    }

get_response = requests.get('https://udamlopsp3.herokuapp.com/')
print(get_response.status_code)
print(get_response.json())

pred = requests.post('https://udamlopsp3.herokuapp.com/inference_main', json=sample)
print(pred.status_code)
print(pred.json())