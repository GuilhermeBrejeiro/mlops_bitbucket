"""
Function to run all the training process and serialize it on a .joblib 

Author: Guilherme Brejeiro
Date: March 13, 2022
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_support_modules.functions.feat_eng import read_data, process_data
from ml_support_modules.functions.ml_model import train_model, predictions
import joblib
import os
import json

# Open the config.json file get the paths variables
with open('../config.json', 'r') as conf:
    config = json.load(conf)
final_data_path = os.path.join(config['output_data_folder'], "final_data.csv")
last_model_path = config['last_model']
train = read_data(final_data_path)


def train_final_data():
    """
    Take the data saved on "final_data" folder and use it to train the model
    Save model and encoder inside "last_model" folder   
    """

    #Create X_train, y_train and the encoder to be used on test sets
    X_train, y_train, encoder = process_data(train, label="quality", training=True, encoder=None)

    #Training the model
    model = train_model(X_train, y_train)

    # Save model
    filename_model = "model_wine_quality.joblib"
    model_path = os.path.join(last_model_path, filename_model)
    joblib.dump(model, model_path)

    # Save StandardScale encoder
    filename_encoder = "encoder.joblib"
    encoder_path = os.path.join(last_model_path, filename_encoder)
    joblib.dump(encoder, encoder_path)

if __name__ == "__main__":
    train_final_data()