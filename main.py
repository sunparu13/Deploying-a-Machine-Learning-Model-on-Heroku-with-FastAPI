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
    education_num: int 
    marital_status: str 
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int 
    capital_loss: int 
    hours_per_week: int 
    native_country: str 

app = FastAPI()

@app.get("/")
async def get_items():
    return {"message": "greeting"}

@app.post("/inference_main")
async def inference_main(input: FeatureConfig):
    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"]
    input = input.dict(by_alias=True)
    df = pd.DataFrame(data=input, index=[0])
    #cat_features_reedit = [feature.replace(
     #   '-', '_') for feature in cat_features]
    #df.rename(columns=cat_features_reedit, inplace=True)
    model, encoder, lb = load_model('model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl' )
    X_test, _, _, _ = process_data(df, categorical_features=cat_features,
                              training=False, label=None, encoder=encoder, lb=lb)
    y_pred = inference(model, X_test)
    if y_pred[0]:
        pred = {"salary": ">50k"}
    else:
        pred = {"salary": "<=50k"}
    return pred
