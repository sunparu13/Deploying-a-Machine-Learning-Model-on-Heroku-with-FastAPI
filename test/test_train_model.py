import os
from starter.train_model import preprocessing, train

def test_preprocessing():
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    X_train, _, _, _ = preprocessing("data/census_clean.csv", cat_features)

    assert X_train.shape[0] > 0
    assert X_train.shape[1] == 12
    assert os.path.isfile("model/encoder.pkl")
    assert os.path.isfile("model/lb.pkl")

def test_train():
    X_train, y_train, _, _ = preprocessing("data/census_clean.csv")
    train(X_train, y_train)
    assert os.path.isfile("model/model.pkl")
    
