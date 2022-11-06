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
        os.system("dvc config core.hardlink_lock true")
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class FeatureConfig(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


app = FastAPI()

model, encoder, lb = load_model(
        'model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl')


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model, encoder, lb = load_model(
        'model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl')


@app.get("/")
async def get_items():
    return {"message": "greeting"}


@app.post("/inference_main")
async def inference_main(input: FeatureConfig):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    input = input.dict(by_alias=True)
    df = pd.DataFrame(data=input, index=[0])
    X_test, _, _, _ = process_data(df, categorical_features=cat_features,
                                   training=False, label=None, encoder=encoder, lb=lb)
    y_pred = inference(model, X_test)
    if y_pred[0]:
        pred = {"salary": "<=50k"}
    else:
        pred = {"salary": ">50k"}
    return pred
