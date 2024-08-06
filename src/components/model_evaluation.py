import os
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error
from urllib.parse import urlparse

from src.logger.logging import logging
from src. exception.exception import CustomException

import mlflow
import mlflow.sklearn
import numpy as np
import pickle
#from src.D

# Data ingestion -> Data Transformation -> Model build/trainer -> model evaluation
# In components we have Configuation and Artifact , Artifact is the output of the component
# In configuration we configure the following data , train.csv, test.csv and raw data.csv where to save the data etc configuration is done in configure
# The Output is saved into Artifacts foloder and from Artifact folder we take artifacts to next stage


class ModelEvaluationConfig:
    pass

class ModelEvaluation():

    def __init__(self):
        pass

    def iniate_model_evaluation(self):
        try:
            pass

        except Exception as e:
            logging.info()
            raise CustomException (e,sys)