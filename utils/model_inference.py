import argparse
import glob
import json

import pandas as pd

from utils import ml_common as mc


def latest_selection(date_str):
    """The model selection in force on date_str (latest training on or before it), or None."""
    paths = [p for p in glob.glob("model_bank/selection_*.json") if mc.date_of(p) <= date_str]
    if not paths:
        return None
    with open(max(paths, key=mc.date_of)) as f:
        return json.load(f)


def main(date_str):
    features = mc.read_partition("datamart/gold/feature_store", "gold_feature_store", date_str)
    selection = latest_selection(date_str)
    if features is None or selection is None:
        print("no applications or no model yet on", date_str, "- skipped")
        return

    frames = []
    for role in ("champion", "challenger"):
        artefact = mc.load_artefact(selection[role])
        out = features[["Customer_ID"]].copy()
        out["snapshot_date"] = date_str
        out["model_id"] = artefact["model_id"]
        out["model_role"] = role
        out["default_probability"] = mc.predict_pd(artefact, features)
        frames.append(out)
        print(role, artefact["model_id"], "mean PD", round(out["default_probability"].mean(), 4))
    mc.write_partition(pd.concat(frames, ignore_index=True), "datamart/gold/model_predictions", "model_predictions",
                       date_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score this month's applications")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
