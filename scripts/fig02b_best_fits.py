"""Figure 2B: best-fit GFP trajectory of each optimizer vs the experimental data.

Output: figures/fig02b.png  (Overleaf: figures/fig02b.png, Figure 2 panel B)
Data:   data/fitting_targets_and_references/experimental_data.mat
        data/fitting_targets_and_references/<optimizer>_best_params.csv
"""

import numpy as np
import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style()
    exp_trace = c.load_experimental_trace()
    time_min = np.arange(0, c.SIM_DURATION_MIN + c.SAMPLE_INTERVAL_MIN, c.SAMPLE_INTERVAL_MIN)

    sims = {algo: c.simulate_gfp(c.load_best_params(algo)) for algo in c.ALGORITHMS}
    rmse = {algo: float(np.sqrt(np.mean((exp_trace - sim) ** 2))) for algo, sim in sims.items()}
    order = sorted(c.ALGORITHMS, key=rmse.get)  # legend lists best fit first

    fig, ax = plt.subplots(figsize=c.FIG_SIZE, constrained_layout=True)
    ax.plot(time_min, exp_trace, "o", color="black", alpha=0.75, markersize=4,
            label="Experimental")
    for algo in order:
        ax.plot(time_min, sims[algo], color=c.COLOR_MAP[algo], linewidth=1.2,
                label=c.DISPLAY_NAMES[algo])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("GFP (a.u.)")
    ax.set_title("Best Model Fits vs Experimental Data")
    ax.set_xlim(0, 490)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_ylim(0, max(sim.max() for sim in sims.values()) * 1.08)
    ax.legend(loc="upper right", frameon=True)
    c.save_figure(fig, "fig02b")


if __name__ == "__main__":
    main()
