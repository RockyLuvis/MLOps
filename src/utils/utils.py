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

        with open(file_path, "wb") as file_obj
            pickle.dump(obj, file_obj)

    except: 
        raise CustomException(e, sys)

def eva
