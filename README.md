<p align="center"><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/en/thumb/2/20/MercadoLibre.svg/1200px-MercadoLibre.svg.png"></p>

# Machine Learning Engineer Evaluation Exam

## Summary

For this challenge I used a Platform as a Service (PaaS) known as Heroku, simulating an environment that receives data into buckets and trains the model with it, if it's performance is better to the performance of the previously trained model, this becomes the model in production, otherwise the previous model remains as the productive model.
P.S. To focus on design and implementation, the dataset and the ML training chosen was as simple as possible
## Achievements
1. Choose a simple database, split it into small slots to simulate the real enviroment that receives new data and retrains the model
2. Develop functions to:
    * Appending new data and retrain the model (preparing.py)
    * Using StandardScaler to preprocessing data and train the model on AdaBoostClassifier (training.py)
    * Scoring model on test data and save f1 score on a .txt file (scoring_data.py)
    * Send the model to production environment (model_deployment.py)
3. Develop an API to get the features and return the prediction from the best scoring model (main.py on heroku)
4. Load testing the API using locust (load_test.py)
5. Create a bitbucket pipeline to automate the process of install dependences, test modules and push to heroku at every commit

![Pipeline](images/full_cicle.png)

## Folders

* data 
    - all_data: gathers all databases that are inputed in the process
    - final_data: all data joined in one big dataset, excluding any duplicate
    - test_data: a small dataset split from the original data, used for test model's predictions
* model
    - last_model: the last model trained, last encoder trained and it's scores on test data
    - production_model: the best model trained, it's encoder and it's scores on test data
* modules
    - ml_support_modules: auxiliar functions used for read, clean and select models
    - preparing.py: take all data available in "all_data", merge it, drop duplicates and save it on "final_data"
    - training.py: train the model using "final_data" content and save model, encoder and logs on "last_model"
    - scoring_data.py: take the model from "last_model" and score it's predictions
    - model_deployment.py: compare the model from 'last_model" with the model from "production_model" the best one become the production model
    - full_cicle.py: run the complete pipeline with all the previously functions

