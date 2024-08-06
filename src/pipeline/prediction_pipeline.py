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
        print("Init the object")

    def predict(self, features):

        try:
            # Sequence of Prediction Pipeline - Data is needed for prediction, Data can be Bulk or single instance(single value prediction)
            # Pass data to Preprocessing before passing data to Model as Transformation technique is needed
            # Pass the data to model
            # Evaluate the result prediction, if bulk data is used prediction can be done. 
            # Evaluation is comparing Predicted value with True value (y, y)

            # collect preprocessor obj from the path
            # Preprocessor.pkl and Model.pkl are available in artifacts. 

            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            #load the preprocessor and Model
            preprocessor = load_object(preprocessor_path)

            print(f"Prediction_pipeline :: preprocessor type: {type(preprocessor)}")

            print("Prediction_pipeline :: model_path", model_path)
            model = load_object(model_path)

            print ("features:",features)
            #Transform the data with preprocessor as per features
            scaled_feature = preprocessor.transform(features)

            print ("scaled_feature:",scaled_feature)
            print(f" Prediction_pipeline :: Model type: {type(model)}")

            #Now pass the transformed data to model for prediction
            prediction = model.predict(scaled_feature)

            logging.info(f"Final Prediction is {prediction}")

            return prediction
    
        except Exception as e:

            raise CustomException (e, sys)
        

# From where the features list witll come? we will need another class
class CustomData:

    def __init__(self, carat:float, depth:float, table:float, x:float, y:float, z:float, cut:str, color:str, clarity: str) -> None:
        #collect the inputs coming from the form.html
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

        #collect the data as an arry or Dataframe


    def get_data_as_dataframe (self):
        # this function arranges the data in the dataframe which will be passed to the model for prediction
        # Basically generate a Feature list to be passed to the Model.
        # Pass the dictionary to pd.dataframe class we can convert this Dictionary into a Data frame
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info (f"Data frame is created from the App incoming data {df}")

            return df
        except Exception as e:
            logging.info ("Exception :: Creation of Dataframe from incoming app data failed")
            raise CustomException (e, sys)

        

