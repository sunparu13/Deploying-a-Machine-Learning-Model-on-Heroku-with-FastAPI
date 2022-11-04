import requests
import json

lower_sample = {
    "age": 53,
    "workclass": "Private",
    "fnlgt": 234721,
    "education": "11th",
    "education_num": 7,
    "marital_status": "Married-civ-spouse",
    "occupation": "Handlers-cleaners",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

pred = requests.post('https://udamlopsp3.herokuapp.com/inference_main', json= lower_sample)
print(pred.status_code)
print(pred)

higher_sample = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

pred = requests.post('https://udamlopsp3.herokuapp.com/inference_main', data=json.dumps(higher_sample))
print(pred.status_code)
print(pred)