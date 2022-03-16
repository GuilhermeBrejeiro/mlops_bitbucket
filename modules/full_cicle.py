"""
It executes the complete cicle of check if new data are introduced inside data/all_data
If it find new data, append it to the rest of the data
Train the model with this new data
Check if the model has a better f1 score on the test data inside data/test_data
If it has a better score, copy the model, encoder and last_score from last_model to production_model
"""


# Import libraries
import pandas as pd
import preparing, training, scoring_data, model_deployment
import os
import json
from sys import exit


def full_pipeline():
    # Open the config.json file get the paths variables
    with open('./config.json', 'r') as conf:
        config = json.load(conf)

    # Path to the model's production data 
    input_data_folder = config["input_data_folder"]
    output_data_folder = config["output_data_folder"]
    last_model = config["last_model"]
    test_data = config['test_data']
    production_model = config["production_model"]


    ###################### Check for new data ##########################
    # Reading how many files composes the production data
    with open(os.path.join(output_data_folder, "data_logs.txt"), "r") as logs:
        all_data_logs = logs.read().splitlines()

    all_data_len = len(os.listdir(input_data_folder))

    if len(all_data_logs) == all_data_len:
        
        return None
    else:
        preparing.merge_and_deduplicate(input_data_folder, output_data_folder)

    ###################### Ingest new data ############################
    # Training model with new data and compare the scores
    with open(os.path.join(production_model, "last_score.txt"), "r") as f:
        prod_score = f.read()

    # The training always send it's output to last_model folder
    training.train_final_data(output_data_folder, last_model)
    # Run scoring_data will generate new scoring on last_model folder
    scoring_data.scoring_model(test_data, last_model)

    with open(os.path.join(last_model, "last_score.txt"), "r") as g:
        new_score = g.read()

    # If we get a better score this become the prod model
    if new_score > prod_score:
        model_deployment.deploy_to_production(last_model, production_model)

    else:
        return None

if __name__ == "__main__":
    full_pipeline()

