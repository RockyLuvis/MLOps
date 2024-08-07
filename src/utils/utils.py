#Write Helper methods here
import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.logger.logging import logging
from src.exception.exception import CustomException

#3 metrcis
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#To save pipeline object or preprocessing obj or transsformation obj or model

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            #convert obj to a physical file
            pickle.dump(obj, file_obj)

    except Exception as e: 
        raise CustomException (e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
            #print(f"Model type: {type(obj)}")
            return obj
    except Exception as e:
        logging.info('Exception occured in load_object function')
        raise CustomException (e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

        for i in range(len(list(models))):
            #Get models
            model = list(models.values())[i]
            #Peform trainig with the model
            model.fit (x_train, y_train)

            #predict with x_test
            y_test_pred = model.predict (x_test)
            
            #Get R2 score
            
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        logging.info ('Exception occured during model training')
        raise CustomException(e, sys)