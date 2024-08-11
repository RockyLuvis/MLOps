import os
import sys

import pandas as pd
import numpy as np

from src.utils.utils import load_object
from src.components.data_ingestion import DataIngestion
from src.exception.exception import CustomException

from src.logger.logging import logging

class BulkPredictConfigClass:
    #configure the paths for bulk file 
    # Configure path for bulk prediction result *.csv
    Prediction_data_path:str= os.path.join ("artifacts", "raw.csv")
    pass

class BulkPredct:

    def __init__(self) -> None:
        pass

    def initiate_bulk_prediction(self, bulk_dir_path ):

        # Load the preprocessor.pkl 
        # Load the Model.pkl 
        # This method will iterate over the bulk file directory for each file and
        # Capture each prediction in a DF and convert that to a result.csv
        # Keep the result.csv in artifacts folder
        model_path = os.path.join ("artifacts", "model.pkl")
        logging.info ("Prediction_pipeline :: model_path", model_path)
        model = load_object(model_path)

        preprocessor_path = os.path.join ("artifacts", "preprocessor.pkl")
        preprocessor = load_object(preprocessor_path)

        for file_name in os.listdir(bulk_dir_path):

            # Call Data ingestion
            if file_name.endswith ('.csv'):

                try:

                    #Read csv and put into a df
                    file_path = os.path.join (bulk_dir_path, file_name)
                    df = pd.read_csv (file_path)

                    #Feature set
                    drop_col = ['price', 'id']
                    features = df.drop (columns=drop_col, axis=1)

                    scaled_features = preprocessor.transform (features)

                    #Predict for scaled features
                    prediction = model.predict(scaled_features)

                    logging.info(f"Final Prediction is {prediction}")

                    df['prediction'] = prediction
                    prediction_data_path:str= os.path.join ("artifacts", f'Bulk_prediction_{file_name}')
                    df.to_csv(prediction_data_path, index=False)
                
                except CustomException as e:
                    logging.info ("Exception occured in initiate_bulk_prediction ")
                    raise Exception (e, sys)


                




