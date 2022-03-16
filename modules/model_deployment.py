"""
This module will just copy files from last model to
production model, will be used everytime a new model
is trained and have a better score than the actual
production model

Author: Guilherme Brejeiro
Date: March 14, 2022
"""
import shutil, os
import json
os.listdir("./")
with open('./modules/config.json', 'r') as conf:
    config = json.load(conf)
# List all files on last_model folder
files = os.listdir(config['last_model']) 
prod_env = config['production_model']
def deploy_to_production():
    """
    Copy encoder.joblib, last_score.txt and model_wine_quality.joblib
    from "last_model" to the production environment
    """
    for file in files:
        full_filename = os.path.join(config['last_model'], file)
        shutil.copy(full_filename, prod_env)


if __name__ == "__main__":
    deploy_to_production()

