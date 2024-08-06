import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.logger.logging import logging
from src. exception.exception import CustomException

from dataclasses import dataclass

#For Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
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
    preprocessor_obj_file_path: str = os.path.join ('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
       self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation(self):
        try:
            logging.info('Data transformation initiated')

            cat_cols = ['cut', 'color', 'clarity']
            num_cols = ['carat','depth','table','x','y','z']

            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preproc_obj = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_cols),
                    ('cat', categorical_transformer, cat_cols)
                ]
            )

            logging.info('Data transformation object created successfully')
            return preproc_obj



        except Exception as e:
            #logging.info()
            raise CustomException(e,sys)
    
    def initialize_data_transformation (self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info ("Read train and test data complete")
            logging.info (f"Train Dataframe Head : \n {train_df.head().to_string()}")
            logging.info (f"Test Dataframe Head : \n {test_df.head().to_string()}")

            preproc_obj = self.get_data_transformation()

            target_col = 'price'
            drop_col = ['price', 'id']

            input_feature_train_df = train_df.drop (columns=drop_col, axis=1)
            target_feature_train_df = train_df[target_col]

            logging.info (f"input_feature_train_df Dataframe Head : \n {input_feature_train_df.head().to_string()}")
            logging.info (f"target_feature_train_df Dataframe Head : \n {target_feature_train_df.head().to_string()}")

            input_feature_test_df = test_df.drop (columns=drop_col, axis=1)
            target_feature_test_df = test_df[target_col]

            input_feature_train_arr = preproc_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preproc_obj.transform(input_feature_test_df)

            logging.info ("Applying preprocessing objects on training and testing data sets")

            #Perform a concatenation of input feature and target feature to pass to the Model as we should have both 
            #independent and dependent features

             # Ensure the dimensions match before concatenation
            if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
                raise CustomException("Mismatch in training data dimensions")

            if input_feature_test_arr.shape[0] != target_feature_test_df.shape[0]:
                raise CustomException("Mismatch in test data dimensions")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #Save the object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preproc_obj
            )

            logging.info ("Preprocessing Pickle file saved")

            return (
                train_arr, test_arr
            )
                
        except Exception as e:
            #logging.info("Exception from Data transformation")
            raise CustomException(e,sys)


