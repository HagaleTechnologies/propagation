from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from propagation.eval.metrics import brier_score, log_loss_score, reliability_bins


def write_headline_report(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, out_dir: Path
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    brier = brier_score(y_true, y_prob)
    logloss = log_loss_score(y_true, y_prob)
    bins = reliability_bins(y_true, y_prob)

    table_path = out_dir / "headline_table.csv"
    write_header = not table_path.exists()
    with open(table_path, "a") as f:
        if write_header:
            f.write("model,brier,log_loss,n\n")
        f.write(f"{model_name},{brier:.6f},{logloss:.6f},{len(y_true)}\n")

    fig, ax = plt.subplots(figsize=(5, 5))
    predicted = [b["mean_predicted"] for b in bins if b["n"]]
    observed = [b["observed_rate"] for b in bins if b["n"]]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    if predicted:
        ax.plot(predicted, observed, marker="o", label=model_name)
    ax.set_xlabel("mean predicted P(open)")
    ax.set_ylabel("observed open rate")
    ax.set_title(f"Reliability diagram — {model_name}")
    ax.legend()
    fig.savefig(out_dir / f"reliability_{model_name}.png", dpi=150)
    plt.close(fig)

    return {"brier": brier, "log_loss": logloss, "bins": bins}
