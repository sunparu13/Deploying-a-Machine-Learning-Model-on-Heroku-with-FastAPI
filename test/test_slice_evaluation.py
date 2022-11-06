import os
from starter.ml.model import slice_evaluation
from starter.train_model import preprocessing, train


def test_slice_evaluation(df, model, cat_features, encoder, lb):
    """
    computes performance on model slices.
    
    Inputs
    ----------
    df: test dataframe
    model: trained model
    cat_features: categorical features
    encoder: encoded dataframe
    lb: encoded label
    Returns
    -------
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    X_train, y_train, encoder, lb, df_test = preprocessing("data/census_clean.csv", cat_features)
    model = train(X_train, y_train)
    slice_evaluation(df_test, model, cat_features, encoder, lb)
    assert os.path.isfile("model/slice_output.txt")



