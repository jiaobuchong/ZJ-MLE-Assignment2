from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# the project folder is mounted here by docker-compose
RUN = "cd /opt/airflow/project && python3 data_pipeline.py"
DATE = ' --snapshotdate "{{ ds }}"'

default_args = {
    "owner": "airflow",
    "depends_on_past": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "loan_default_data_pipeline",
    default_args=default_args,
    description="Monthly datamart: bronze -> silver -> gold label store and feature store",
    schedule_interval="0 0 1 * *", # run on the start of the month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),  # same months as the ML pipeline
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,  # starts backfilling as soon as Airflow is up; the ML pipeline waits on it
) as dag:

    start = EmptyOperator(task_id="start")

    label_store = BashOperator(task_id="label_store", bash_command=RUN + " --store label" + DATE)
    feature_store = BashOperator(task_id="feature_store", bash_command=RUN + " --store feature" + DATE)

    # the ML pipeline's sensor waits for this task
    datamart_ready = EmptyOperator(task_id="datamart_ready")

    start >> [label_store, feature_store] >> datamart_ready
