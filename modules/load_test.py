from locust import HttpUser, task
import json
class SimpleUser(HttpUser):
    @task
    def get_predict(self):

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


        self.client.post("/predict", data=json.dumps(api_parameters))
