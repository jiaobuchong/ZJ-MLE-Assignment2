import glob
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score


def add_months(date_str, months):
    date = datetime.strptime(date_str, "%Y-%m-%d") + relativedelta(months=months)
    return date.strftime("%Y-%m-%d")


def date_of(path):
    # .../gold_label_store_2024_01_01.parquet -> 2024-01-01
    return "-".join(os.path.basename(path).split(".")[0].split("_")[-3:])


# ---------- gold tables (one parquet per month) ----------

def partition_path(directory, prefix, date_str):
    # gold_feature_store + 2024-01-01 -> <directory>/gold_feature_store_2024_01_01.parquet
    return os.path.join(directory, prefix + "_" + date_str.replace("-", "_") + ".parquet")


def read_partition(directory, prefix, date_str):
    """One month as pandas, or None if it was not written."""
    path = partition_path(directory, prefix, date_str)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df if len(df) > 0 else None


def read_partitions(directory, prefix, date_to):
    """All months up to date_to stacked, or an empty frame."""
    paths = sorted(glob.glob(os.path.join(directory, prefix + "_*.parquet")))
    frames = [pd.read_parquet(p) for p in paths if date_of(p) <= date_to]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_partition(df, directory, prefix, date_str):
    os.makedirs(directory, exist_ok=True)
    path = partition_path(directory, prefix, date_str)
    df.to_parquet(path, index=False)
    print("saved to:", path, "rows:", len(df))


def label_months_available(date_str):
    """Label store months that exist on date_str (no peeking at later months)."""
    paths = glob.glob("datamart/gold/label_store/gold_label_store_*.parquet")
    return sorted(d for d in map(date_of, paths) if d <= date_str)


# ---------- model artefacts ----------

def artefact_path(model_id):
    return os.path.join("model_bank", model_id + ".pkl")


def load_artefact(model_id):
    with open(artefact_path(model_id), "rb") as f:
        return pickle.load(f)


def as_matrix(df, feature_cols):
    return df.reindex(columns=feature_cols).astype(float).replace([np.inf, -np.inf], np.nan)


def predict_pd(artefact, df):
    """Probability of default per row."""
    return artefact["model"].predict_proba(as_matrix(df, artefact["feature_cols"]))[:, 1]


# ---------- metrics ----------

def auc_gini(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    auc = roc_auc_score(y_true, y_score)
    return auc, 2 * auc - 1


def make_reference(scores, n_bins=10):
    """Decile edges and the share of scores per decile, kept with the model for PSI."""
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1))[1:-1])
    return {"edges": edges.tolist(), "expected": bin_shares(scores, edges).tolist()}


def bin_shares(scores, edges):
    counts = np.bincount(np.searchsorted(edges, scores, side="right"), minlength=len(edges) + 1)
    return counts / len(scores)


def psi(reference, scores):
    """Population stability index of scores against the model's reference deciles."""
    expected = np.clip(np.asarray(reference["expected"]), 1e-4, None)
    actual = np.clip(bin_shares(scores, np.asarray(reference["edges"])), 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
