# Script to train machine learning model.

# Add the necessary imports for the starter code.
import sys
sys.path.append('/Users/sunparu/Documents/schulung/mldevops/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/data')
sys.path.append('/Users/sunparu/Documents/schulung/mldevops/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/model')
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
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
    return X_train, y_train, encoder, lb

# train model
def train(X_train, y_train):
    model = train_model(X_train, y_train)
    with open("model/model.pkl", 'wb') as file_model:
        pickle.dump(model, file_model)

#if __name__ == "__main__":
 #   cat_features = [
 #   "workclass",
 #   "education",
 #   "marital-status",
 #   "occupation",
 #   "relationship",
 #   "race",
 #   "sex",
 #   "native-country"]

 #   X_train, y_train, encoder, lb = preprocessing("data/census_clean.csv", cat_features)
 #   train(X_train, y_train)

