import os
import sys

from src.logger.logging import logging
from src.exception.exception import CustomException

import pandas as pd
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
#
# Stage 1 - Data Ingestion
#

#obj = DataIngestion()

#train_data_path, test_data_path = obj.initiate_data_ingestion()

#Stage 2 - Data Transformation
#
#data_transformation = DataTransformation()

#train_arr, test_arr= data_transformation.initialize_data_transformation (train_data_path, test_data_path)

#Stage 3 - Train the model
# Train the Model
#model_trainer_obj = ModelTrainer()
#model_trainer_obj.initiate_model_training(train_arr, test_arr)

#Stage 4 - Model evaluation
#Evaluate the Model
#model_eval_obj = ModelEvaluation()
#model_eval_obj.iniate_model_evaluation(train_arr, test_arr)

class TrainingPipeline:

    def self():
        pass

    def ingestdata():
        try:
            DataIngestionObj = DataIngestion()
            train_data_path, test_data_path = DataIngestionObj.initiate_data_ingestion()
            return train_data_path, test_data_path

        except Exception as e:
            logging.info(" Exception occurred during Data Ingestion in Training Pipeline")
            raise CustomException (e, sys)

    def TransformData(self, train_data_path, test_data_path ):
        try:
            TransformDataObj = DataTransformation()
            train_arr, test_arr = TransformDataObj.initialize_data_transformation(train_data_path, test_data_path) 
            return train_arr, test_arr
        
        except Exception as e:
            logging.info("Exception occurred in Data Transformation in the training pipeline")
            raise CustomException (e, sys)
        
    def TrainModel(self, train_arr, test_arr ):
        try:
            ModelTrainerObj = ModelTrainer()
            ModelTrainer.initiate_model_training(train_arr, test_arr)
        except CustomException as e:
            logging.info("Exception occurred during Train Model")
            raise CustomException (e, sys)
        
    def EvaluateModel(self, train_array, test_array):
        try:
            ModelEvaluationObj = ModelEvaluation()
            ModelEvaluationObj.iniate_model_evaluation(train_array, test_array)
        
        except CustomException as e:
            logging.info("Exception during Model evaluation")
            raise Exception (e, sys)
        
    # Call them in Sequence 
    def start_training(self):
        try:
            train_data_path, test_data_path = self.ingestdata()
            train_arr, test_arr = self.TransformData(train_data_path, test_data_path )
            self.TrainModel(train_arr, test_arr)
        except CustomException as e:
            logging.info ("Exception occurred during Train Model")
            raise Exception (e, sys)
        






