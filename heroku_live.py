import requests

lower_sample = {
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

pred = requests.post('https://udamlopsp3.herokuapp.com/inference_main', json=lower_sample)
print(pred.status_code)
print(pred.json())

higher_sample = {
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
}

pred = requests.post('https://udamlopsp3.herokuapp.com/inference_main', json=higher_sample)
print(pred.status_code)
print(pred.json())