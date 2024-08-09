import os
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse

from src.logger.logging import logging
from src.exception.exception import CustomException

import mlflow
import mlflow.sklearn
import numpy as np
import pickle

from src.utils.utils import load_object 


# Data ingestion -> Data Transformation -> Model build/trainer -> model evaluation
# In components we have Configuation and Artifact , Artifact is the output of the component
# In configuration we configure the following data , train.csv, test.csv and raw data.csv where to save the data etc configuration is done in configure
# The Output is saved into Artifacts foloder and from Artifact folder we take artifacts to next stage


class ModelEvaluationConfig:

    pass

class ModelEvaluation():

    def __init__(self):
        logging.info("Starting Evaluation of the model....")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score (actual, pred)

        logging.info(f"C evaluation metrics {rmse}, {mae}, {r2} - rmse, mae, r2")
        return rmse, mae, r2


    def iniate_model_evaluation(self, train_array, test_array):
        try:
            #Segreegate Test and train data.
            x_test, y_test = (test_array[:,:-1],
                              test_array[:,-1])
            x_train, y_train = (train_array[:,:-1],
                                train_array[:,-1]
                                )
            
            model_path = os.path.join ("artifacts", "model.pkl")
            model = load_object(model_path)

            #Specify path where to register the Model
            #pass the url here
            mlflow.set_registry_uri("./mystore/")
            logging.info(" Model has been registred ")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(x_test)
                (rmse, mae, r2) = self.eval_metrics(y_test, prediction)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2",r2)

                #if we logging in local and not in a registry_uri then use this condition
                if tracking_url_type_store != "file":

                    #Register the model
                    #If cloud location is avilable log in cloud location. 
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    #Log locally
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logging.info("Exception in Model Evaluation")
            raise CustomException (e,sys)