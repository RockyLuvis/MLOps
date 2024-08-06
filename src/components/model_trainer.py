import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from src.logger.logging import logging
from src. exception.exception import CustomException
from dataclasses import dataclass

from src.utils.utils import save_object, evaluate_model

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error #For evaluation
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join ('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_training(self, train_array, test_array):

        try:
            logging.info ('Splitting dependent and Independent variables')
            #x_train, y_train, x_test, y_test = (
            #    train_array[:,:-1],
            #    train_array[:,:-1],
            #    test_array[:,:-1],
            #    test_array[:,:-1],
            #)

            x_train = train_array[:, :-1]  # Features
            y_train = train_array[:, -1]   # Target
            x_test = test_array[:, :-1]    # Features
            y_test = test_array[:, -1] 

            logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                #'Randomforest':RandomForestRegressor(),
                #'xgboost': XGBRegressor()
            }
            #evaluate_model(x_train, y_train, x_test, y_test, models):
            model_report:dict= evaluate_model (x_train, y_train, x_test, y_test, models)
            logging.info (model_report)
            print (f'model_report : {model_report}')

            best_model_score = max (model_report.values())

            #best_model_name = list(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
    

            print (f"Best Model found, Model name : {best_model_name}")
            print ('\n=============================')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_name
            )
            

        except Exception as e:
            logging.info("Exception from model Trainer")
            raise CustomException (e,sys)