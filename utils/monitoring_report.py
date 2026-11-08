import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils import ml_common as mc  # noqa: E402

REPORT_DIR = "datamart/gold/monitoring_report"
ROLE_COLOURS = {"champion": "#2a78d6", "challenger": "#eb6834"}  # same colour per role in every chart
WATCH_COLOUR, ALERT_COLOUR = "#fab219", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.labelcolor": "#52514e", "legend.frameon": False, "font.size": 10,
})


def line_chart(df, value_col, title, subtitle, ylabel, thresholds, train_dates, filename):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.subplots_adjust(top=0.78, left=0.07, right=0.97, bottom=0.12)
    fig.text(0.07, 0.95, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.07, 0.895, subtitle, fontsize=10, color="#52514e", va="top")

    for role, colour in ROLE_COLOURS.items():
        # one line per model, so the old and the retrained model are not joined
        for i, (model_id, seg) in enumerate(df[df["model_role"] == role].groupby("model_id")):
            seg = seg.sort_values("app_month")
            ax.plot(pd.to_datetime(seg["app_month"]), seg[value_col], color=colour, linewidth=2, marker="o",
                    markersize=4, label=role if i == 0 else None)
    for value, colour, text in thresholds:
        ax.axhline(value, color=colour, linewidth=1.2, linestyle=(0, (4, 3)), label=text)
    for date in train_dates:
        ax.axvline(pd.Timestamp(date), color="#c3c2b7", linewidth=1, zorder=0)
        ax.annotate("models trained " + date, (pd.Timestamp(date), 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", fontsize=8.5, color="#52514e", va="top")

    ax.set_ylim(0, max([df[value_col].max()] + [t[0] for t in thresholds]) * 1.2)  # room for the labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Application month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, borderaxespad=0)
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)


def main(date_str):
    monitoring = mc.read_partitions("datamart/gold/model_monitoring", "model_monitoring", date_to=date_str)
    if len(monitoring) == 0:
        print("no monitoring results yet on", date_str, "- skipped")
        return
    os.makedirs(REPORT_DIR, exist_ok=True)
    monitoring.to_csv(os.path.join(REPORT_DIR, "monitoring_summary.csv"), index=False)
    train_dates = sorted(mc.date_of(p) for p in glob.glob("model_bank/selection_*.json") if mc.date_of(p) <= date_str)

    performance = monitoring[monitoring["check"] == "performance"]
    if len(performance) > 0:
        line_chart(performance, "gini", "Model performance by application month",
                   "Gini on loans that reached 6 months on book (label: 30+ DPD). "
                   "Champion makes the decision; challenger runs in shadow.",
                   "Gini (2 x AUC - 1)", [], train_dates, "01_model_performance_gini.png")

    stability = monitoring[monitoring["check"] == "stability"]
    line_chart(stability, "psi", "Score stability by application month",
               "PSI of this month's scores vs the model's holdout scores. Below 0.10 stable, 0.25+ retrain.",
               "Score PSI", [(0.10, WATCH_COLOUR, "watch 0.10"), (0.25, ALERT_COLOUR, "alert 0.25")],
               train_dates, "02_score_stability_psi.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitoring charts")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
