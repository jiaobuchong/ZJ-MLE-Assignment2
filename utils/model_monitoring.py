import argparse

import pandas as pd

from utils import ml_common as mc

# thresholds from the SOP
GINI_DROP_ALERT = 0.10  # Gini this far below the training OOT Gini: retrain
PSI_WATCH = 0.10        # score PSI 0.10 - 0.25: watch
PSI_ALERT = 0.25        # score PSI >= 0.25: population shifted, retrain


def performance_rows(date_str):
    app_month = mc.add_months(date_str, -6)  # labels are 30+ DPD at 6 months on book
    labels = mc.read_partition("datamart/gold/label_store", "gold_label_store", date_str)
    predictions = mc.read_partition("datamart/gold/model_predictions", "model_predictions", app_month)
    if labels is None or predictions is None:
        return []
    rows = []
    for (model_id, role), group in predictions.groupby(["model_id", "model_role"]):
        df = group.merge(labels[["Customer_ID", "label"]], on="Customer_ID")
        auc, gini = mc.auc_gini(df["label"], df["default_probability"])
        oot_gini = mc.load_artefact(model_id)["results"]["gini_oot"]
        rows.append({"check": "performance", "app_month": app_month, "model_id": model_id, "model_role": role,
                     "n": len(df), "auc": auc, "gini": gini, "default_rate": df["label"].mean(),
                     "mean_pd": df["default_probability"].mean(),
                     "status": "alert" if gini < oot_gini - GINI_DROP_ALERT else "ok"})
    return rows


def stability_rows(date_str):
    predictions = mc.read_partition("datamart/gold/model_predictions", "model_predictions", date_str)
    if predictions is None:
        return []
    rows = []
    for (model_id, role), group in predictions.groupby(["model_id", "model_role"]):
        value = mc.psi(mc.load_artefact(model_id)["score_reference"], group["default_probability"].values)
        status = "alert" if value >= PSI_ALERT else "watch" if value >= PSI_WATCH else "ok"
        rows.append({"check": "stability", "app_month": date_str, "model_id": model_id, "model_role": role,
                     "n": len(group), "psi": value, "mean_pd": group["default_probability"].mean(), "status": status})
    return rows


def main(date_str):
    rows = performance_rows(date_str) + stability_rows(date_str)
    if not rows:
        print("nothing to monitor on", date_str)
        return
    df = pd.DataFrame(rows)
    df.insert(0, "monitor_date", date_str)
    mc.write_partition(df, "datamart/gold/model_monitoring", "model_monitoring", date_str)
    print(df.drop(columns="monitor_date").round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monthly model monitoring")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
