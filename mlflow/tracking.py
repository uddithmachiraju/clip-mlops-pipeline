import json 
import mlflow 
import mlflow.pytorch 

with open("src/data/parameters.json", "r") as file:
    parameters = json.load(file) 

class Tracking:
    def __init__(self, experiment_name = "CLIP-mlops"):
        mlflow.set_tracking_uri("http://localhost:5000") 
        mlflow.set_experiment(experiment_name) 

    def log_params(self, params):
        """Log the parameters"""
        with mlflow.start_run():
            mlflow.log_params(params) 

    def log_metrics(self, metrics, step = None):
        """Log the metrics"""
        with mlflow.start_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step = step) 

    def log_model(self, model, model_name = 'clip-model'):
        """Log the model"""
        with mlflow.start_run():
            mlflow.pytorch.log_model(model, model_name) 

if __name__ == "__main__":
    tracker = Tracking()
    tracker.log_params(
        parameters
    )