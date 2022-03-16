from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import os
import json
import joblib
from modules.ml_support_modules.functions.feat_eng import process_data


app = FastAPI()
os.listdir("./")

# Open the config.json file get the paths variables
with open('./modules/config.json', 'r') as conf:
    config = json.load(conf)
# Load model
model_path = os.path.join(config['production_model'], "model_wine_quality.joblib")
model = joblib.load(model_path)
# Load encoder
encoder_path = os.path.join(config['production_model'], "encoder.joblib")
encoder = joblib.load(encoder_path)


@app.get("/")
def welcome():

    return "MeLi project"

class Input(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.70)
    citric_acid: float = Field(..., example=0.04)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=15.0)
    total_sulfur_dioxide: float = Field(..., example=54.0)
    density: float = Field(..., example=0.9968)
    pH: float = Field(..., example=3.16)
    sulphates: float = Field(..., example=0.58)
    alcohol: float = Field(..., example=9.4)

@app.post("/predict/")
async def wine_quality_prediction(item:Input):
    """
    Receive arguments from Input class and predict the quality of the wine
    """

    input_dict = {
    'fixed_acidity': [item.fixed_acidity],
    'volatile_acidity': [item.volatile_acidity],
    'citric_acid': [item.citric_acid],
    'residual_sugar': [item.residual_sugar],
    'chlorides': [item.chlorides],
    'free_sulfur_dioxide': [item.free_sulfur_dioxide],
    'total_sulfur_dioxide': [item.total_sulfur_dioxide],
    'density': [item.density],
    'pH': [item.pH],
    'sulphates': [item.sulphates],
    'alcohol': [item.alcohol]
    }

    print(input_dict)
    # Create a DataFrame using the values from the dict
    df = pd.DataFrame.from_dict(input_dict)
    print(df)

    X, _, _ = process_data(
                            df, 
                            label=None,
                            training=False, 
                            encoder=encoder
                            )

    y_pred = model.predict(X)

    if y_pred[0] is True:
        answer = "Good quality wine!"
    else:
        answer = "Bad quality wine!"
    return answer
