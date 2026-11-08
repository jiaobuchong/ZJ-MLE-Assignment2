import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import ml_common as mc

RANDOM_STATE = 88

# Clickstream is excluded: applicants from Jul 2024 on have none, so a model trained with it
# mis-scores them (PSI > 2, see analysis/run_with_clickstream). Remove this line to use it again.
EXCLUDED_FEATURE_PREFIXES = ("fe_", "clickstream_months")


def build_dataset(train_date):
    """Labelled applications known on train_date, with a split column: train / test / oot."""
    frames = []
    for label_month in mc.label_months_available(train_date)[-12:]:  # latest 12 months of labels
        app_month = mc.add_months(label_month, -6)  # label = 30+ DPD at 6 months on book
        labels = mc.read_partition("datamart/gold/label_store", "gold_label_store", label_month)
        features = mc.read_partition("datamart/gold/feature_store", "gold_feature_store", app_month)
        df = labels[["Customer_ID", "label"]].merge(features.drop(columns=["snapshot_date"]), on="Customer_ID")
        df["app_month"] = app_month
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    oot_months = sorted(data["app_month"].unique())[-2:]  # latest 2 application months
    data["split"] = np.where(data["app_month"].isin(oot_months), "oot", "train")
    in_time = data.index[data["split"] == "train"]
    _, test_idx = train_test_split(in_time, test_size=0.2, random_state=RANDOM_STATE,
                                   stratify=data.loc[in_time, "label"])
    data.loc[test_idx, "split"] = "test"
    return data


def train_xgboost(X, y):
    # trees split on raw values and learn where missing values go: no imputing or scaling
    search = RandomizedSearchCV(
        xgb.XGBClassifier(eval_metric="auc", tree_method="hist", n_jobs=1, random_state=RANDOM_STATE),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],  # shallow trees: only about 3-6k training rows
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.9],
            "min_child_weight": [1, 5, 10],
            "reg_lambda": [1, 5, 10],
        },
        n_iter=30, cv=3, random_state=RANDOM_STATE, n_jobs=4,
        scoring="roc_auc",  # AUC on probabilities (make_scorer(roc_auc_score) would score 0/1 predictions)
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_logreg(X, y):
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000)),  # L1 drops weak features
    ])
    search = GridSearchCV(pipeline, {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]}, cv=3, scoring="roc_auc", n_jobs=4)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def main(algorithm, train_date):
    data = build_dataset(train_date)
    skip = {"Customer_ID", "label", "app_month", "split"}
    feature_cols = [c for c in data.columns
                    if c not in skip and not c.startswith(EXCLUDED_FEATURE_PREFIXES)]
    split = {s: data[data["split"] == s] for s in ("train", "test", "oot")}
    for s, df in split.items():
        print(s, len(df), "rows, default rate", round(df["label"].mean(), 3))

    train = split["train"]
    trainer = train_xgboost if algorithm == "xgboost" else train_logreg
    model, params = trainer(mc.as_matrix(train, feature_cols), train["label"])

    results, scores = {}, {}
    for s, df in split.items():
        scores[s] = model.predict_proba(mc.as_matrix(df, feature_cols))[:, 1]
        auc, gini = mc.auc_gini(df["label"], scores[s])
        results["auc_" + s], results["gini_" + s] = round(auc, 4), round(gini, 4)
    print(algorithm, results)

    model_id = algorithm + "_" + train_date.replace("-", "_")
    artefact = {
        "model_id": model_id,
        "algorithm": algorithm,
        "model": model,
        "feature_cols": feature_cols,
        "train_date": train_date,
        "app_months": sorted(str(m) for m in data["app_month"].unique()),
        "rows": {s: len(df) for s, df in split.items()},
        "results": results,
        "hp_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
        # score deciles on the holdout (test + OOT), the baseline for the monitoring PSI
        "score_reference": mc.make_reference(np.concatenate([scores["test"], scores["oot"]])),
    }
    os.makedirs("model_bank", exist_ok=True)
    with open(mc.artefact_path(model_id), "wb") as f:
        pickle.dump(artefact, f)
    with open(os.path.join("model_bank", model_id + ".json"), "w") as f:
        json.dump({k: v for k, v in artefact.items() if k not in ("model", "score_reference")}, f, indent=2)
    print("saved to:", mc.artefact_path(model_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train one candidate model")
    parser.add_argument("--algorithm", choices=["xgboost", "logreg"], required=True)
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD, the training date")
    args = parser.parse_args()
    main(args.algorithm, args.snapshotdate)
