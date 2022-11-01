import pickle
import logging
from numpy import mean
from numpy import std

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def load_model(model_path, encoder_path, lb_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_encoder = pickle.load(open(encoder_path, 'rb'))
    loaded_lb = pickle.load(open(lb_path, 'rb'))
    return loaded_model, loaded_encoder, loaded_lb

def slice_generation(df, model, cat_features, encoder, lb):
    """
    computes performance on model slices.
    
    Inputs
    ----------
    model: trained model
    cat_features
    encoder
    lb
    Returns
    -------
    """
    for feature in cat_features:
        for cls in df[feature].unique():
            slice = df[df[feature] == cls]
            x_test, y_test, _, _ = process_data(
                    slice,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)
            y_pred_slice = inference(model, slice)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred_slice)
