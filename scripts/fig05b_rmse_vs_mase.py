"""Figure 5B: fit (RMSE) vs parameter recovery (MASE) for each synthetic run.

As in the paper version, runs on an optimizer's own dataset family are excluded
(600 of the 750 runs), the same filter as Figure 5A. Note: the current Overleaf
caption says N = 750.

Output: figures/fig05b.png  (Overleaf: figures/fig05b.png, Figure 5 panel B)
Data:   data/synthetic_results.csv
        data/fitting_targets_and_references/<family>_best_params.csv (ground truth)
"""

import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()
    df = df[df["optimizer"] != df["category"]].copy()

    fig, ax = plt.subplots(figsize=c.FIG_SIZE, constrained_layout=True)
    for algo in c.ALGORITHMS:
        runs = df[df["optimizer"] == algo]
        ax.scatter(runs["rmse"], runs["mase"], s=10, color=c.COLOR_MAP[algo],
                   alpha=0.5, linewidths=0)
    for algo in c.ALGORITHMS:
        runs = df[df["optimizer"] == algo]
        ax.scatter(runs["rmse"].median(), runs["mase"].median(), marker="D", s=24,
                   color=c.COLOR_MAP[algo], edgecolors="black", linewidths=0.6,
                   label=c.DISPLAY_NAMES[algo], zorder=3)

    ax.set_xlabel("RMSE")
    ax.set_ylabel("MASE")
    ax.set_title("Fit vs Recovery")
    ax.legend(loc="upper right", frameon=True, fontsize=7, handletextpad=0.3,
              labelspacing=0.25, borderpad=0.3)
    c.save_figure(fig, "fig05b")


if __name__ == "__main__":
    main()
