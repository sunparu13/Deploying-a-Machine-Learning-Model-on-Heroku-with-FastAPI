# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, load_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class FeatureConfig(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

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


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "greeting"}


@app.post("/inference_main")
async def inference_main(features: FeatureConfig):
    df = pd.DataFrame(data=features.dict(by_alias=True), index=[0])
    df = df['age',
      'workclass',
      'fnlgt',
      'education',
      'education-num',
      'marital-status',
      'occupation',
      'relationship',
      'race',
      'sex',
      'capital-gain',
      'capital-loss',
      'hours-per-week',
      'native-country'
    ]
    cat_features_reedit = [feature.replace(
        '-', '_') for feature in cat_features]
    model, encoder, lb = load_model('model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl' )
    X_test, _, _, _ = process_data(df, categorical_features=cat_features_reedit,
                              training=False, label=None, encoder=encoder, lb=lb)
    y_pred = inference(model, X_test)
    prediction = lb.inverse_transform(y_pred)[0]
    print(prediction)
    return {"prediction": prediction}
