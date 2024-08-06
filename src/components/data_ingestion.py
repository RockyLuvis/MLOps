import os
import sys
from pathlib import Path

import pandas as pd
import numpy as py
sys.path.insert(0, os.path.abspath('src'))
from src.logger.logging import logging
from src.exception.exception import CustomException

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
class DataIngestionConfig:
        raw_data_path:str= os.path.join ("artifacts", "raw.csv")
        train_data_path:str= os.path.join ("artifacts", "train.csv")
        test_data_path:str= os.path.join ("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
         self.ingestion_config = DataIngestionConfig()

         
    def initiate_data_ingestion(self):
        try:
            logging.info("Started Data Ingestion")
            data = pd.read_csv("./playground-series-s3e8/train.csv")
            logging.info ( "reading a df" )

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" Saved the data in the artifact folder ")

            #Split the test and train data
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("train test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion part completed")

            return( 
                 self.ingestion_config.test_data_path,
                 self.ingestion_config.train_data_path
            )
        

        except Exception as e:
            logging.info()
            raise CustomException(e,sys)

if __name__ == "__main__":
     Dataobj = DataIngestion()
     Dataobj.initiate_data_ingestion()
