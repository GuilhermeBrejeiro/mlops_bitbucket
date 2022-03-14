"""
Function to run all the training process and serialize it on a .joblib 

Author: Guilherme Brejeiro
Date: March 13, 2022
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_support_modules.functions.feat_eng import read_data, process_data
from ml_support_modules.functions.ml_model import train_model, acc_f1_metrics, predictions
import joblib
import os

# Import data
data_folder = os.path.abspath("../data/")
path = os.path.join(data_folder, "WineQT.csv")
data = read_data(path)

# Split data into train and test
train, test = train_test_split(data, test_size=0.2)

#Create X_train, y_train and the encoder to be used on test sets
X_train, y_train, encoder = process_data(train, label="quality", training=True, encoder=None)

#Creating X_test, y_test using the encoder created previously
X_test, y_test, _ = process_data(test, label="quality", training=False, encoder=encoder)

#Training the model
model = train_model(X_train, y_train)

#Making prediction on test set
y_pred = predictions(model, X_test)

# Model folder path
filename_path = os.path.abspath("../model/")

# Save model
filename_model = "model_wine_quality.joblib"
model_path = os.path.join(filename_path, filename_model)
joblib.dump(model, model_path)

# Save StandardScale encoder
filename_encoder = "encoder.joblib"
encoder_path = os.path.join(filename_path, filename_encoder)
joblib.dump(encoder, encoder_path)
