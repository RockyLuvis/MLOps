import os
import sys

import pandas as pd
import numpy as np

from src.exception.exception import CustomException
from src.logger.logging import logging

# We need load method from util to load pkl files
from src.utils.utils import load_object

class PredictPipeline:

    def __init__(self) -> None:
        pass

    def predict(self, features):

        try:
            # Sequence of Prediction Pipeline - Data is needed for prediction, Data can be Bulk or single instance(single value prediction)
            # Pass data to Preprocessing before passing data to Model as Transformation technique is needed
            # Pass the data to model
            # Evaluate the result prediction, if bulk data is used prediction can be done. 
            # Evaluation is comparing Predicted value with True value (y, y)

            # collect preprocessor obj from the path
            # Preprocessor.pkl and Model.pkl are available in artifacts. 

            preprocessor_path = os.path.join("artifacts", "preprocesor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            #load the preprocessor and Model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            #Transform the data with preprocessor as per features
            scaled_feature = preprocessor.transform()

            #Now pass the transformed data to model for prediction
            prediction = model.predict(scaled_feature)

            return prediction
    
        except Exception as e:
            raise CustomException (e, sys)
        

# From where the features list witll come? we will need another class
class CustomData:

    def __init__(self) -> None:
        pass

    def get_data_as_dataframe (self):
        # this function arranges the data in the dataframe which will be passed to the model for prediction
        # Basically generate a Feature list to be passed to the Model.
        pass

