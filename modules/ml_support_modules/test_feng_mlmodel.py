"""
Functions to test the ml_model.py
"""

import os
import pandas as pd
import pytest
#Sklearn modules
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
#Developed modules
from functions.feat_eng import read_data, process_data
from functions.ml_model import train_model, acc_f1_metrics, predictions


data_folder = os.path.abspath("./mlops/modules/ml_support_modules/WineQT.csv")

@pytest.fixture
def data_read():
    """
    Read data to use on the tests
    """
    path = os.path.join(data_folder, "test_data.csv")
    df = read_data(path)
    return df

def test_read_data(data_read):
    """
    Testing if the data was reading correctly based on it's shape
    """
    assert data_read.shape[0] > 0
    assert data_read.shape[1] > 1

def test_process_data(data_read):
    """
    Test if drop of columns happened and if the len of X and y are equal
    """

    features = ['fixed_acidity', 
                'volatile_acidity', 
                'citric_acid', 
                'residual_sugar',
                'chlorides', 
                'free_sulfur_dioxide', 
                'total_sulfur_dioxide', 
                'density',
                'pH', 
                'sulphates', 
                'alcohol'
                ]

    X, y, encoder = process_data(data_read, label="quality", training=True)

    assert len(X[0]) == len(features)
    assert len(X) == len(y)

def test_train_pred_metrics(data_read):
    """
    Test the complete cicle of train the model, test data into the model and compute metrics
    """

    #Split data into train and test
    train, test = train_test_split(data_read, test_size=0.2)

    #Create X_train, y_train and the encoder to be used on test sets
    X_train, y_train, encoder = process_data(train, label="quality", training=True, encoder=None)

    #Creating X_test, y_test using the encoder created previously
    X_test, y_test, encoder = process_data(test, label="quality", training=False, encoder=encoder)

    #Training the model
    model = train_model(X_train, y_train)

    #Making prediction on test set
    y_pred = predictions(model, X_test)

    #Testing if the metrics are not empty
    accuracy, f1 = acc_f1_metrics(y_test, y_pred)

    assert accuracy > 0
    assert f1 > 0
    

