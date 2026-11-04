"""Figure 2A: best RMSE of each of the 13 runs per optimizer on the experimental data.

Output: figures/fig02a.png  (Overleaf: figures/fig02a.png, Figure 2 panel A)
Data:   data/experimental_results/*_history_variant_AAV_run_*.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import common as c


def main() -> None:
    c.apply_publication_style()
    num_points = len(c.load_experimental_trace())
    rows = [
        {"algorithm": c.DISPLAY_NAMES[algo], "rmse": np.sqrt(hist["sse"].min() / num_points)}
        for algo, _, hist in c.iter_run_histories()
    ]
    df = pd.DataFrame(rows)

    medians = df.groupby("algorithm")["rmse"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=c.FIG_SIZE, constrained_layout=True)
    sns.boxplot(ax=ax, data=df, x="algorithm", y="rmse", order=order,
                color="lightgray", width=0.30, fliersize=0)
    sns.swarmplot(ax=ax, data=df, x="algorithm", y="rmse", order=order,
                  hue="algorithm", palette=c.DISPLAY_COLOR_MAP, size=3, linewidth=0)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("RMSE (a.u.)")
    ax.set_title(f"RMSE across {c.NUM_RUNS} runs ({c.VARIANT})")
    c.style_optimizer_xticks(ax, order)

    for i, algo in enumerate(order):
        m = float(medians[algo])
        ax.text(i + 0.15, m, f"{m:.2f}", ha="left", va="center", fontsize=8, color="black")

    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, float(df["rmse"].max()) * 1.15)
    c.save_figure(fig, "fig02a")


if __name__ == "__main__":
    main()
