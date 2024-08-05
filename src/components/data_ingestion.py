import os
import sys
from pathlib import Path

import pandas as pd
import numpy as py

from src.logger.logging import logging
from src. exception.exception import CustomException

from dataclasses import dataclass

#For Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#First stage : Training of the mode hence create a pipeline for training
#second stage : Prediction, create a seperate pipeline for prediction
#For training : Data ingestion -> Data Transformation -> Build Data model -> Evaluate
#Prediction : can be Bulk or single value prediction , Data -> Transformation , use the same object (scaling related object or encdoing related obj or any other) that was used for training
# we will compose all objects in the same Pipepine, save them in preprocessor obj and use it in transformation
# After transformation do the prediction
#

from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e,sys)

