# Script to train machine learning model.
# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, load_model, inference, compute_model_metrics
import pickle


def preprocessing(data_path, cat_features):
    """
    Load the data set and split it into train and test subset
    Args:
        data_path: path of the dataset
    Returns:
        df_train: train dataframe
        df_test: test dataframe
    """
    data = pd.read_csv(data_path)
    df_train, df_test = train_test_split(data, test_size=0.2)

    X_train, y_train, encoder, lb = process_data(
        df_train, cat_features, label="salary", training=True
    )

    with open("model/encoder.pkl", 'wb') as file_encoder:
        pickle.dump(encoder, file_encoder)
    with open("model/lb.pkl", 'wb') as file_lb:
        pickle.dump(lb, file_lb)
    return X_train, y_train, encoder, lb, df_test

# train model
def train(X_train, y_train):
    model = train_model(X_train, y_train)
    with open("model/model.pkl", 'wb') as file_model:
        pickle.dump(model, file_model)

def test_model(test, cat_features=None, label="salary"):    
    # load model
    model, encoder, lb = load_model('model/model.pkl', 'model/encoder.pkl', 'model/lb.pkl' )    
    # testing
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    return precision, recall, fbeta


if __name__ == "__main__":
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
    train(X_train, y_train)
    p,r,f = test_model(df_test, cat_features, label="salary" )
