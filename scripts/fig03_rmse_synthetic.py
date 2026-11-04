"""Figure 3: fitting accuracy (RMSE) of each optimizer over all 750 synthetic runs.

Output: figures/fig03.png  (Overleaf: figures/fig03.png, Figure 3)
Data:   data/synthetic_results.csv
"""

import matplotlib.pyplot as plt
import seaborn as sns

import common as c


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()

    medians = df.groupby("algorithm")["rmse"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=c.FIG_SIZE, constrained_layout=True)
    sns.boxplot(ax=ax, data=df, x="algorithm", y="rmse", order=order,
                color="lightgray", width=0.30, fliersize=0)
    sns.swarmplot(ax=ax, data=df, x="algorithm", y="rmse", order=order,
                  hue="algorithm", palette=c.DISPLAY_COLOR_MAP, size=1.5, linewidth=0)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("RMSE (a.u.)")
    ax.set_title("Overall Optimizer Performance")
    c.style_optimizer_xticks(ax, order)

    for i, algo in enumerate(order):
        m = float(medians[algo])
        ax.text(i + 0.15, m, f"{m:.2f}", ha="left", va="center", fontsize=8, color="black")

    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, float(df["rmse"].max()) * 1.15)
    c.save_figure(fig, "fig03")


if __name__ == "__main__":
    main()
