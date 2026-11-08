import argparse
import json
import os

from utils import ml_common as mc


def main(train_date):
    suffix = train_date.replace("-", "_")
    candidates = [mc.load_artefact(algorithm + "_" + suffix) for algorithm in ("xgboost", "logreg")]
    candidates.sort(key=lambda a: a["results"]["gini_oot"], reverse=True)
    champion, challenger = candidates[0], candidates[1]

    selection = {
        "train_date": train_date,
        "champion": champion["model_id"],
        "challenger": challenger["model_id"],
        "rule": "highest out-of-time Gini becomes champion",
        "oot_gini": {a["model_id"]: a["results"]["gini_oot"] for a in candidates},
    }
    path = os.path.join("model_bank", "selection_" + suffix + ".json")
    with open(path, "w") as f:
        json.dump(selection, f, indent=2)
    print(json.dumps(selection, indent=2))
    print("saved to:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="choose champion and challenger")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD, the training date")
    args = parser.parse_args()
    main(args.snapshotdate)
