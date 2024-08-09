# This file creates the data_ingestion_task >> data_transform_task >> model_trainer_task flow in DAG
import pendulum
from airflow import DAG
from airflow import PythonOperator

import json
from textwrap import dedent

from src.pipeline.training_pipeline import Training_pipeline
TrainingPipelineObj = Training_pipeline()

from src.pipeline.training_pipeline import TrainingPipeline

import numpy as np

#Define DAG
with DAG (
    "diamondTrainingPipeline",
    default_args={"retries":2},
    description="This DAG is for Training pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2024,8,9, tz="UTC"),
    catch=False,
    tags=["MLOps" , "classification" , "gemstone"]
) as dag:
    
    dag.doc_md = __doc__
    
    # Stage 1
    def data_ingestion ( **kwargs ):
        ti = kwargs[ "ti" ] 
        train_data_path, test_data_path = TrainingPipeline.ingestdata()
        
        #Push the artifacts to the next stage
        ti.xcom_push("data_ingestion_artifact", {"train_data_path": train_data_path , "test_data_path": test_data_path })
    
    #Stage 2
    def data_transformation ( **kwargs ):
        ti = kwargs [ "ti" ]

        #Pull data_ingestion_artifact
        data_ingestion_artifact = ti.xcom_pull ( task_ids = "data_ingestion", key = "data_ingestion_artifact" )
        train_arr, test_arr = TrainingPipeline.TransformData( data_ingestion_artifact["train_data_path"], data_ingestion_artifact["test_data_path"] )

        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()

        #Push artifacts
        ti.xcom_push ("data_transformation_artifact", {"train_arr":train_arr, "test_arr":test_arr})

    #stage 3
    def model_trainer ( **kwargs ):
        ti = kwargs ["ti"]

        #Pull the arr from data_transformation artifact
        data_transformation_artifact = ti.xcom_pull ( task_ids = "data_transformation", key = "data_transformation_artifact")

        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])

        TrainingPipeline.TrainModel(train_arr, test_arr)
    
    # Stage 4
    # Push to AWS or local artifactory mystorage
    def push_data_to_s3 ( ):
        import os
        app_path = "/app/artifacts"
        print ("Pushing to S3")

        #Save it to S3
        #os.system (f" aws sync {app_app} s3:/{bucket_name}/artifact")
    
    data_ingestion_task = PythonOperator(
        task_id = "data_ingestion",
        python_callable = data_ingestion,
    )

    data_ingestion_task.doc_md = dedent(
        """\ data_ingestion_task :: Ingest the data file and create test and train file"""
    )

    data_transformation_task = PythonOperator(
        task_id = "data_transformation",
        python_callable = data_transformation
    )

    data_transformation_task.doc_md = dedent(
        """\ data_transformation_task:: Performs data transformation and prepares test and train data"""
    )

    data_train_task = PythonOperator(
        task_id = "model_trainer",
        python_callable = model_trainer
    )

    data_train_task.doc_md = dedent (
        """\ data_train_task :: Perform Model training and get best model"""
    )

    push_data_to_s3_task = PythonOperator(
        task_id = "push_data_to_s3",
        python_callable = push_data_to_s3
    )

    #Sequence the tasks
    data_ingestion_task >> data_transformation_task >> data_train_task >> push_data_to_s3_task





