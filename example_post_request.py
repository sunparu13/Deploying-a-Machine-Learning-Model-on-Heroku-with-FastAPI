import json
import requests

response = requests.post(
    "https://127.0.0.1:8000",
    data=json.dumps({
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
    }),
)

print(response)

class Config:
        schema_extra = {
            "example_lower50k": {
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
            },
            "example_over50k": {
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
        }
