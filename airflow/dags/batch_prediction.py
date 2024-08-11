import os
import sys

from airflow import DAG
import pendulum
from airflow.operators.python import PythonOperator
from src.pipeline.bulk_prediction_pipeline import BulkPredict

with DAG (
    "BulkGemStonePrediction_Raveendra",
    default_args={"retries":2},
    description="This DAG predicts gemstone price in bulk"
    schedule_interval="@daily"
    start_date=pendulum.datetime(2024,8,11, tz="UTC"),
    catchup=False,
    tags=['genstone', "MLOps", "classification"]
) as dag:
    DAG.doc_md= __doc__

    #Stage 1
    def downloadBulkDataFiles( **kwargs ):
        ti = kwargs[ti]
        print ("Downloaded data files from S3 bucket")
        dir_path = 'artifacts/train.csv'
        #Push XCOM dir_path
        ti.xcom_push("dir_path",  {"bulk_dir_path": dir_path })
        

    #Stage 2
    def batchpredict( **kwargs ):

        ti = kwargs["ti"]

        #From XCOM pull bulk_dir_path
        bulk_dir_path = ti.xcom_pull(task_ids = "downloadBulkDataFiles", key = "dir_path" )
                
        BulkPredictobj = BulkPredict( )
        BulkPredictobj.initiate_bulk_prediction(bulk_dir_path )
        
    #Stage 3
    def uploadPredictions():
        print ("Upload Result data to S3 bucket")


download_bulk_data_files = PythonOperator (
    task_id = "download_bulk_data_files",
    python_callable= downloadBulkDataFiles
)

batch_predict = PythonOperator (
    task_id = "batch_predict",
    python_callable = batchpredict
)

upload_prediction_results = PythonOperator (
    task_id = "upload_prediction_results",
    python_callable = uploadPredictions
)
download_bulk_data_files >> batch_predict >> upload_prediction_results
