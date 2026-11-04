"""Figure 4: parameter recovery (MASE), all datasets plus one panel per dataset family.

Output: figures/fig04.png  (Overleaf: figures/fig04.png, Figure 4)
Data:   data/synthetic_results.csv
        data/fitting_targets_and_references/<family>_best_params.csv (ground truth)

Note: this figure uses matplotlib's default style (not the 90 x 70 mm publication
style) because that is how the version in the paper was made.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import common as c


def _panel(ax, data, title: str) -> None:
    medians = data.groupby("optimizer")["mase"].median().sort_values()
    order = medians.index.tolist()
    sns.boxplot(ax=ax, data=data, x="optimizer", y="mase", order=order,
                color="lightgray", width=0.3, fliersize=0)
    sns.swarmplot(ax=ax, data=data, x="optimizer", y="mase", hue="optimizer",
                  palette=c.COLOR_MAP, order=order, size=5)
    ax.set_title(title)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("MASE")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.DISPLAY_NAMES[o] for o in order])
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    for i, opt in enumerate(order):
        m = medians.get(opt, np.nan)
        if not np.isnan(m):
            ax.text(i + 0.15, m, f"{m:.3f}", ha="left", va="center", fontsize=10, color="black")


def main() -> None:
    df = c.load_synthetic_results()
    families = sorted(df["category"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    ax_list = axes.flatten()
    _panel(ax_list[0], df, "Overall Parameter Recovery (MASE)")
    for ax, family in zip(ax_list[1:], families):
        _panel(ax, df[df["category"] == family], c.DISPLAY_NAMES[family])
    for ax in ax_list:
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    c.save_figure(fig, "fig04", dpi=300, tight=False)


if __name__ == "__main__":
    main()
