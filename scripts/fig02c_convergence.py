"""Figure 2C: convergence of the best-so-far RMSE (median and IQR over 13 runs).

Output: figures/fig02c.png  (Overleaf: figures/fig02c.png, Figure 2 panel C)
Data:   data/experimental_results/*_history_variant_AAV_run_*.csv
"""

import numpy as np
import matplotlib.pyplot as plt

import common as c

MAX_EVALS = 22000
# First evaluation, then every 2000 evaluations from 1000 to 21000
EVAL_POINTS = np.r_[1, np.arange(1000, 22001, 2000)]
LEGEND_ORDER = ["cmaes", "lshade", "basin_hopping", "dual_annealing", "pso"]


def main() -> None:
    c.apply_publication_style()
    num_points = len(c.load_experimental_trace())

    histories = {algo: [] for algo in c.ALGORITHMS}
    for algo, _, hist in c.iter_run_histories():
        best_so_far = np.minimum.accumulate(np.sqrt(hist["sse"].to_numpy(float) / num_points))
        padded = np.full(MAX_EVALS, np.nan)
        n = min(MAX_EVALS, len(best_so_far))
        padded[:n] = best_so_far[:n]
        histories[algo].append(padded)

    fig, ax = plt.subplots(figsize=c.FIG_SIZE, constrained_layout=True)
    for algo in LEGEND_ORDER:
        sampled = np.vstack(histories[algo])[:, EVAL_POINTS - 1]
        median = np.nanmedian(sampled, axis=0)
        q1, q3 = np.nanpercentile(sampled, [25, 75], axis=0)
        color = c.COLOR_MAP[algo]
        ax.fill_between(EVAL_POINTS, q1, q3, step="post", color=color, alpha=0.16, linewidth=0)
        ax.step(EVAL_POINTS, median, where="post", color=color, linewidth=1.2,
                label=c.DISPLAY_NAMES[algo])
        ax.plot(EVAL_POINTS[-1], median[-1], "o", color=color, markersize=3.5)

    ax.set_title("Convergence of RMSE (median ± IQR)")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("RMSE (a.u.)")
    ax.set_xlim(0, 23000)
    ax.set_ylim(0, 400)
    ax.set_xticks([0, 3000, 9000, 15000, 21000])
    ax.legend(loc="upper right", frameon=True)
    c.save_figure(fig, "fig02c")


if __name__ == "__main__":
    main()
