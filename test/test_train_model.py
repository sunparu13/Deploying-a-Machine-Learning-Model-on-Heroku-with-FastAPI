import os
import sys
sys.path.append('/Users/sunparu/Documents/schulung/mldevops/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/data')
sys.path.append('/Users/sunparu/Documents/schulung/mldevops/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/model')
sys.path.append('/Users/sunparu/Documents/schulung/mldevops/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/starter')
from train_model import preprocessing, train

def test_preprocessing():
    X_train, _, _, _ = preprocessing("data/census_clean.csv")

    assert X_train.shape[0] > 0
    assert X_train.shape[1] == 12
    assert os.path.isfile("model/encoder.pkl")
    assert os.path.isfile("model/lb.pkl")


def test_train(x,y):
    train(x, y)
    assert os.path.isfile("model/model.pkl")
    
