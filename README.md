# Plotting for the optimization paper

Code and data that regenerate every data figure in *A Comparative Study on Different Optimization Solvers for Mechanistic ODE-based Gene Circuit Model Parameterization* (Carden, Tang, An).

Each figure has one script, named after its Overleaf file. `scripts/fig02b_best_fits.py` writes `figures/fig02b.png`, which is `figures/fig02b.png` in Overleaf (Figure 2, panel B), so a regenerated PNG can be uploaded to Overleaf as-is.

## Run

```bash
pip install -r requirements.txt
python make_all_figures.py           # all figures
python make_all_figures.py fig02b    # one figure
```

## Figure map

| Paper figure | Script | Output (= Overleaf file) | Shows |
|---|---|---|---|
| Figure 1 | not in this repo | `fig01.png` | Circuit diagram and ODEs (not a data plot) |
| Figure 2A | `scripts/fig02a_rmse_experimental.py` | `figures/fig02a.png` | Best RMSE of each of 13 runs per optimizer, experimental data |
| Figure 2B | `scripts/fig02b_best_fits.py` | `figures/fig02b.png` | Best-fit GFP curve per optimizer vs measured GFP |
| Figure 2C | `scripts/fig02c_convergence.py` | `figures/fig02c.png` | Best-so-far RMSE vs function evaluations (median ± IQR) |
| Figure 3 | `scripts/fig03_rmse_synthetic.py` | `figures/fig03.png` | RMSE over all 750 synthetic runs |
| Figure 4 | `scripts/fig04_param_recovery.py` | `figures/fig04.png` | Parameter recovery (MASE): all datasets + one panel per dataset family |
| Figure 5A | `scripts/fig05a_param_recovery_excluding_own.py` | `figures/fig05a.png` | MASE with each optimizer's own dataset family removed |
| Figure 5B | `scripts/fig05b_rmse_vs_mase.py` | `figures/fig05b.png` | RMSE vs MASE per synthetic run, with per-optimizer medians |

## Editing a figure

- To change one figure, edit only that figure's script.
- `scripts/common.py` holds what all figures share: optimizer names and colors, the 90 × 70 mm / 600 dpi publication style, data loading, the ODE model, and MASE. Changing it changes every figure.
- Figure 4 uses matplotlib's default style rather than the shared publication style, because that is how the paper version was made.

## Data

| Path | Contents |
|---|---|
| `data/experimental_results/` | 65 run histories (5 optimizers × 13 runs) fitting the AAV experimental data. Each row is one function evaluation: `params`, `sse`, `elapsed_time_s`. |
| `data/fitting_targets_and_references/experimental_data.mat` | Measured GFP for the AAV variant: 43 points, 0–420 min, every 10 min. |
| `data/fitting_targets_and_references/<optimizer>_best_params.csv` | Best experimental fit from each optimizer (21 parameters). Also the ground truth for the synthetic dataset family of the same name. |
| `data/fitting_targets_and_references/<optimizer>_<1-3>.csv` | The 15 synthetic datasets (5 families × 3), each with 3 noisy replicates. |
| `data/synthetic_results.csv` | 750 synthetic runs (5 optimizers × 15 datasets × 10 seeds): `optimizer`, `dataset`, `seed`, `min_sse`, `params`. |

Definitions: RMSE = √(SSE / 43). MASE scales each of the 21 parameters to [0, 1] by its bounds and averages the absolute error against the ground truth.

## Known differences from the Overleaf text

- Figure 5B, like the version in the paper, plots the 600 runs left after removing each optimizer's own dataset family, but the caption says N = 750.
- In the left "Overall" panel of Figure 4, seaborn can't place about 5% of the swarm dots and prints a warning, so those dots are missing; the boxes and median labels still use every run.
