"""
Function to run all the training process and serialize it on a .joblib 

Author: Guilherme Brejeiro
Date: March 14, 2022
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_support_modules.functions.feat_eng import read_data, process_data
from ml_support_modules.functions.ml_model import predictions, acc_f1_metrics
import joblib
import os
import json

# Open the config.json file get the paths variables
with open('../config.json', 'r') as conf:
    config = json.load(conf)
# Load test data
test_data_path = os.path.join(config['test_data'], "test_data.csv")
test = read_data(test_data_path)
# Load model
model_path = os.path.join(config['last_model'], "model_wine_quality.joblib")
model = joblib.load(model_path)
# Load encoder
encoder_path = os.path.join(config['last_model'], "encoder.joblib")
encoder = joblib.load(encoder_path)

def scoring_model():
    """
    Use the data saved on "test_data" folder to evaluate the performance of
    the model saved on "last_model" folder
    Save the scoring on a .txt file inside "last_model" folder
    """
    # Creating X_test, y_test using the encoder created previously
    X_test, y_test, _ = process_data(test, label="quality", training=False, encoder=encoder)

   # Making prediction on test set
    y_pred = predictions(model, X_test)

    # Evaluate
    _, f1 = acc_f1_metrics(y_test, y_pred)

    with open(os.path.join(config['last_model'], 'last_score.txt'), 'w') as lastscore:
        lastscore.write(str(f1))

if __name__ == "__main__":
    scoring_model()




