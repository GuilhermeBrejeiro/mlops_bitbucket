"""
This module will just copy files from last model to
production model, will be used everytime a new model
is trained and have a better score than the actual
production model

Author: Guilherme Brejeiro
Date: March 14, 2022
"""
import shutil, os


def deploy_to_production(last_model, production_model):
    """
    Copy encoder.joblib, last_score.txt and model_wine_quality.joblib
    from "last_model" to the production environment
    """
    files = os.listdir(last_model)

    for file in files:
        full_filename = os.path.join(last_model, file)
        shutil.copy(full_filename, production_model)


if __name__ == "__main__":
    deploy_to_production(last_model, production_model)

