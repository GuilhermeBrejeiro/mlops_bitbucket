"""
Function to send requests to the API
"""
import requests
import json

api_parameters = {
  "fixed_acidity": 7.8,
  "volatile_acidity": 0.76,
  "citric_acid": 0.04,
  "residual_sugar": 2.3,
  "chlorides": 0.092,
  "free_sulfur_dioxide": 15,
  "total_sulfur_dioxide": 54,
  "density": 0.9970,
  "pH": 3.26,
  "sulphates": 0.65,
  "alcohol": 9.8
}

def request_to_api(api_parameters=api_parameters):
  """
  Function to POST and request a prediction from the API
  """

  response = requests.post("https://guilhermebrejeiromeli.herokuapp.com/predict", data=json.dumps(api_parameters))

  return response.elapsed.total_seconds()

if __name__ == "__main__":
  answer = request_to_api()
  print(answer)