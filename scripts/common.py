"""Shared paths, style, data loading, and model code for the paper figures."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTAL_RUNS_DIR = DATA_DIR / "experimental_results"
TARGETS_DIR = DATA_DIR / "fitting_targets_and_references"
SYNTHETIC_RESULTS_CSV = DATA_DIR / "synthetic_results.csv"
FIGURES_DIR = REPO_ROOT / "figures"

VARIANT = "AAV"
NUM_RUNS = 13

# ---------------------------------------------------------------------------
# Optimizer names and colors (shared by every figure)
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "basin_hopping": "Basin Hopping",
    "cmaes": "CMA-ES",
    "dual_annealing": "Dual Annealing",
    "lshade": "L-SHADE",
    "pso": "PSO",
}
ALGORITHMS = list(DISPLAY_NAMES.keys())
COLOR_MAP = {
    "basin_hopping": "#FFBE0B",
    "cmaes": "#FB5607",
    "dual_annealing": "#FF006E",
    "lshade": "#8338EC",
    "pso": "#3A86FF",
}
DISPLAY_COLOR_MAP = {DISPLAY_NAMES[a]: COLOR_MAP[a] for a in ALGORITHMS}

# ---------------------------------------------------------------------------
# Publication style: 90 x 70 mm panels saved at 600 dpi
# ---------------------------------------------------------------------------
MM_TO_INCH = 1.0 / 25.4
FIG_WIDTH_MM = 90
FIG_HEIGHT_MM = 70
FIG_SIZE = (FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM * MM_TO_INCH)
DPI = 600


def apply_publication_style() -> None:
    """Set the rcParams used by the single-panel paper figures."""
    plt.rcParams.update({
        "figure.figsize": FIG_SIZE,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_optimizer_xticks(ax: plt.Axes, labels: list[str]) -> None:
    """Rotate optimizer names on the x-axis the same way in every figure."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    for lab in ax.get_xticklabels():
        lab.set_rotation(32)
        lab.set_horizontalalignment("center")
        lab.set_verticalalignment("top")
        lab.set_rotation_mode("anchor")
    ax.tick_params(axis="x", pad=12)


def save_figure(fig: plt.Figure, name: str, dpi: int = DPI, tight: bool = True) -> Path:
    """Save ``figures/<name>.png`` (name matches the Overleaf file name)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Experimental data and run histories
# ---------------------------------------------------------------------------
def load_experimental_trace() -> np.ndarray:
    """GFP measurements (43 points, 0-420 min every 10 min) for the AAV variant."""
    return loadmat(TARGETS_DIR / "experimental_data.mat")[VARIANT].ravel().astype(float)


def iter_run_histories():
    """Yield (algorithm, run_index, history DataFrame) for all 65 experimental runs."""
    for algo in ALGORITHMS:
        for run_idx in range(1, NUM_RUNS + 1):
            path = EXPERIMENTAL_RUNS_DIR / f"{algo}_history_variant_{VARIANT}_run_{run_idx}.csv"
            if path.exists():
                yield algo, run_idx, pd.read_csv(path, usecols=["sse"])


def load_best_params(algo: str) -> np.ndarray:
    """Best parameter vector found by ``algo`` on the experimental data."""
    df = pd.read_csv(TARGETS_DIR / f"{algo}_best_params.csv")
    return df.iloc[0].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# ODE model (11 states, 21 parameters) used to draw the best fits
# ---------------------------------------------------------------------------
MODEL_PARAMS = {"P_x": 1e-9, "P_y": 1e-9, "P_z": 1e-9}
ATC_CONVERSION = 0.46822
ATC_NG_PER_ML = 50
SIM_DURATION_MIN = 420
SAMPLE_INTERVAL_MIN = 10


def _derivatives(t: float, state: np.ndarray, p: np.ndarray) -> np.ndarray:
    (STAR, THS, TetR, aTc, tetR_aTc, Y, Y_active, Pz_repressed, Pz_active, Z, GFP) = state
    free_pz = MODEL_PARAMS["P_z"] - Pz_repressed - Pz_active
    d = np.zeros(11)
    d[0] = MODEL_PARAMS["P_x"] * p[0] - p[11] * STAR * free_pz - p[5] * STAR + p[12] * Pz_active
    d[1] = MODEL_PARAMS["P_x"] * p[0] - p[13] * THS * Y - p[6] * THS
    d[2] = (
        p[1] * Y_active
        - p[14] * TetR * free_pz
        - p[15] * TetR * aTc
        - p[7] * TetR
        + p[16] * Pz_repressed
        + p[17] * tetR_aTc
        - p[20] * TetR * Pz_active
    )
    d[3] = -p[15] * TetR * aTc + p[17] * tetR_aTc
    d[4] = p[15] * TetR * aTc - p[17] * tetR_aTc
    d[5] = p[2] * MODEL_PARAMS["P_y"] - p[13] * THS * Y - p[8] * Y
    d[6] = p[13] * THS * Y - p[19] * Y_active
    d[7] = p[14] * TetR * free_pz - p[16] * Pz_repressed + p[20] * TetR * Pz_active
    d[8] = p[11] * STAR * free_pz - p[12] * Pz_active - p[20] * TetR * Pz_active
    d[9] = p[3] * Pz_active - p[9] * Z
    d[10] = p[4] * Z - p[10] * GFP
    return d


def simulate_gfp(params: np.ndarray) -> np.ndarray:
    """Simulated GFP fluorescence (a.u.) at the 43 measurement times."""
    p = np.asarray(params, dtype=float).copy()
    p[7] *= 2  # AAV tag doubles TetR degradation
    state0 = np.zeros(11)
    state0[3] = (ATC_NG_PER_ML / ATC_CONVERSION) * 1e-9
    t_eval = np.arange(0, SIM_DURATION_MIN + SAMPLE_INTERVAL_MIN, SAMPLE_INTERVAL_MIN) * 60.0
    sol = solve_ivp(
        lambda t, y: _derivatives(t, y, p),
        [t_eval[0], t_eval[-1]],
        state0,
        t_eval=t_eval,
        method="BDF",
        atol=1e-11,
        rtol=1e-11,
    )
    return sol.y[10] * 1e9 * 10 ** p[18]


# ---------------------------------------------------------------------------
# Synthetic benchmark (750 runs) and parameter recovery (MASE)
# ---------------------------------------------------------------------------
LOWER_BOUNDS = np.array([
    0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
    0.0001, 0.0001, 0.00005, 0.0001, 0.0001, 0.00005,
    100, 1e-8, 100, 100, 100, 1e-8, 1e-8, 0.5, 0.0001, 100,
])
UPPER_BOUNDS = np.array([
    0.5, 0.5, 0.5, 0.5, 0.5,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    1_000_000, 1e-5, 1_000_000, 1_000_000, 1_000_000, 1e-5, 1e-5, 5, 0.01, 1_000_000,
])


def synthetic_dataset_length() -> int:
    """Number of time points in each synthetic dataset (43)."""
    return len(pd.read_csv(TARGETS_DIR / "basin_hopping_1.csv"))


def mase(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error after scaling every parameter to [0, 1] by its bounds."""
    scale = UPPER_BOUNDS - LOWER_BOUNDS
    return float(np.mean(np.abs((found - LOWER_BOUNDS) / scale - (truth - LOWER_BOUNDS) / scale)))


def load_synthetic_results() -> pd.DataFrame:
    """All 750 synthetic runs with RMSE, MASE, and dataset family ("category")."""
    df = pd.read_csv(SYNTHETIC_RESULTS_CSV)
    df["category"] = df["dataset"].str.rsplit("_", n=1).str[0]
    df["rmse"] = (df["min_sse"] / synthetic_dataset_length()) ** 0.5
    truth = {c: load_best_params(c) for c in df["category"].unique()}
    df["mase"] = [
        mase(np.asarray(ast.literal_eval(params), dtype=float), truth[cat])
        for params, cat in zip(df["params"], df["category"])
    ]
    df["algorithm"] = df["optimizer"].map(DISPLAY_NAMES)
    return df
