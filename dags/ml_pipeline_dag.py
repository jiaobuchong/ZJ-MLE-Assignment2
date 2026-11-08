from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.sensors.external_task import ExternalTaskSensor

# SOP: first model once 8 months of labels exist, then refresh every 6 months
TRAIN_DATES = ["2024-02-01", "2024-08-01"]

# the project folder is mounted here by docker-compose
RUN = "cd /opt/airflow/project && python3 -m utils."
DATE = ' --snapshotdate "{{ ds }}"'

default_args = {
    "owner": "airflow",
    "depends_on_past": True,  # months run in order: a month's models come from earlier months
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "loan_default_ml_pipeline",
    default_args=default_args,
    description="Monthly ML: training, inference, monitoring",
    schedule_interval="0 0 1 * *",  # same schedule as the data pipeline, so the sensor matches months
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),  # last month with new labels
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=True,  # nothing runs until the DAG is turned on in the UI
) as dag:

    # reschedule mode frees the worker slot between checks; with the SequentialExecutor
    # a waiting sensor would otherwise block the data pipeline it waits for
    wait_for_datamart = ExternalTaskSensor(
        task_id="wait_for_datamart",
        external_dag_id="loan_default_data_pipeline",
        external_task_id="datamart_ready",
        failed_states=["failed", "upstream_failed"],
        mode="reschedule",
        poke_interval=20,
        timeout=6 * 60 * 60,
    )

    # other months: the training tasks are skipped, inference uses the latest models
    is_training_month = ShortCircuitOperator(
        task_id="is_training_month",
        python_callable=lambda ds: ds in TRAIN_DATES,
        ignore_downstream_trigger_rules=False,
    )
    model_train_xgboost = BashOperator(task_id="model_train_xgboost",
                                       bash_command=RUN + "model_train --algorithm xgboost" + DATE)
    model_train_logreg = BashOperator(task_id="model_train_logreg",
                                      bash_command=RUN + "model_train --algorithm logreg" + DATE)
    model_select = BashOperator(task_id="model_select", bash_command=RUN + "model_select" + DATE)

    model_inference = BashOperator(task_id="model_inference", bash_command=RUN + "model_inference" + DATE,
                                   trigger_rule="none_failed")  # runs also when training was skipped
    model_monitoring = BashOperator(task_id="model_monitoring", bash_command=RUN + "model_monitoring" + DATE)
    monitoring_report = BashOperator(task_id="monitoring_report", bash_command=RUN + "monitoring_report" + DATE)

    wait_for_datamart >> is_training_month >> [model_train_xgboost, model_train_logreg] >> model_select
    model_select >> model_inference >> model_monitoring >> monitoring_report
