"""Figure 5A: parameter recovery (MASE) with each optimizer's own dataset family removed.

Output: figures/fig05a.png  (Overleaf: figures/fig05a.png, Figure 5 panel A)
Data:   data/synthetic_results.csv
        data/fitting_targets_and_references/<family>_best_params.csv (ground truth)
"""

import matplotlib.pyplot as plt
import seaborn as sns

import common as c


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()
    df = df[df["optimizer"] != df["category"]].copy()

    medians = df.groupby("algorithm")["mase"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=c.FIG_SIZE, constrained_layout=True)
    sns.boxplot(ax=ax, data=df, x="algorithm", y="mase", order=order,
                color="lightgray", width=0.30, fliersize=0)
    sns.swarmplot(ax=ax, data=df, x="algorithm", y="mase", order=order,
                  hue="algorithm", palette=c.DISPLAY_COLOR_MAP, size=1.5, linewidth=0)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("MASE")
    ax.set_title("Parameter Recovery (Excluding Own Dataset)")
    c.style_optimizer_xticks(ax, order)

    for i, algo in enumerate(order):
        m = float(medians[algo])
        ax.text(i + 0.15, m, f"{m:.3f}", ha="left", va="center", fontsize=8, color="black")

    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, float(df["mase"].max()) * 1.15)
    c.save_figure(fig, "fig05a")


if __name__ == "__main__":
    main()
