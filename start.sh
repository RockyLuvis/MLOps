#!bin/sh

# Activate Airflow schedule and webserver
echo "++++++++++++++++++++++++++++++++++++++++++"
echo "Activating Airflow schedule and webserver"
echo "++++++++++++++++++++++++++++++++++++++++++"
# Initialize the Airflow database
airflow db upgrade

echo "Python path:" $PYTHONPATH

airflow users create -e abc@gmail.com -f Harry -l Potter -p raveendra -r Admin -u harry

nohup airflow scheduler > /app/airflow/airflow-scheduler.log 2>&1 & airflow webserver