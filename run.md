## How to run

### 1. Start the services

```bash
docker compose up --build
```

Once the web UI answers (about a minute), the output shows the address and login:

```
airflow-ready-1  | Airflow is ready:  http://localhost:8080
airflow-ready-1  | Username: admin    Password: admin
```

To run in the background, use `docker compose up -d --build`; the same lines are then in `docker compose logs airflow-ready`.

Everything below is done in the Airflow UI.

### 2. Run the pipelines

`loan_default_data_pipeline` starts on its own once Airflow is up. `loan_default_ml_pipeline` is paused: on the DAGs page, turn it on with the toggle on the left (or click ▶ Trigger). Each one backfills every month from 2023-01-01 to 2025-07-01 (catchup). The whole run takes about 12 minutes. Open a DAG and use the Grid view to follow the progress.

1. `loan_default_data_pipeline` builds the datamart for each month: bronze, silver, then the gold label store and feature store.
2. `loan_default_ml_pipeline` waits for the same month of the data pipeline, then trains, scores and monitors. Models are trained on 2024-02-01 and 2024-08-01, and the training tasks are skipped in other months. Each time, XGBoost and logistic regression are trained; the one with the higher OOT Gini becomes `champion` and the other `challenger`. Both score every month from 2024-02 on.

The ML pipeline only starts a month after the data pipeline has finished that month, so it may wait a little at the start.

To stop the backfill part-way, turn the DAGs off again. Months that already finished are kept, and turning the DAGs back on continues from where they stopped.

### 3. Check the outputs

- Models: `model_bank/*.pkl`, champion / challenger in `model_bank/selection_*.json`
- Predictions: `datamart/gold/model_predictions/`
- Monitoring (AUC, Gini, PSI): `datamart/gold/model_monitoring/`
- Charts: `datamart/gold/monitoring_report/*.png`