#!bin/sh

# Activate Airflow schedule and webserver
nohub airflow scheduler &
airflow webserver