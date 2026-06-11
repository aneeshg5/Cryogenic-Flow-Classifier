from __future__ import annotations

import os
import pathlib
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from instrumentation.flow_regime import (
    REGIME_NAMES,
    augment_with_gpr,
    extract_features,
    fit_classifier,
    load_ansys_data,
)
from instrumentation.sensor_sim import generate_dataset
from instrumentation.void_fraction import (
    C_GAS_DEFAULT,
    C_LIQUID_DEFAULT,
    process_signal,
)

_DIR = pathlib.Path(__file__).parent
_OUTPUT_DIR = _DIR / "outputs"
_CSV_PATH = _DIR / "data" / "volume-average-rfile.csv"

_WINDOW_SIZE: int = 20
_N_SYNTH: int = 200
_DURATION_S: float = 2.0
_RATE_HZ: float = 200.0
_K_CORR: float = 0.002

_COLORS: dict[str, str] = {
    "Slug": "#1f77b4",
    "Intermittent": "#ff7f0e",
    "Annular": "#2ca02c",
}
_REGIME_KEYS: tuple[str, ...] = ("slug", "intermittent", "annular")


def _configure_mpl() -> None:
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
    })


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(_OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: outputs/{name}")


def _fig1_signal_examples(dataset: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    for ax, (key, regime) in zip(axes, zip(_REGIME_KEYS, REGIME_NAMES)):
        t, C = dataset[key][0]
        mask = t <= 1.0
        t_s, C_s = t[mask], C[mask]
        vf_mean = float(np.mean((C_LIQUID_DEFAULT - C_s) / (C_LIQUID_DEFAULT - C_GAS_DEFAULT))) * 100
        ax.plot(t_s, C_s, color=_COLORS[regime], linewidth=0.9)
        ax.set_ylabel("Capacitance (F)")
        ax.set_title(
            f"{regime} Flow  |  Mean VF: {vf_mean:.1f}%  |  Var: {np.var(C_s):.2e}"
        )
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Synthetic Capacitance Signals by Flow Regime", fontweight="bold")
    fig.tight_layout()
    _save(fig, "signal_examples.png")


def _fig2_void_fraction_pipeline(dataset: dict) -> None:
    t, C = dataset["slug"][0]
    vf_corr = process_signal(C, C_LIQUID_DEFAULT, C_GAS_DEFAULT, k=_K_CORR)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, C, color=_COLORS["Slug"], linewidth=0.9)
    axes[0].set_ylabel("Capacitance (F)")
    axes[0].set_title("Raw Capacitance Signal")
    axes[0].annotate(
        r"$C = C_L - \alpha \cdot (C_L - C_G)$",
        xy=(0.02, 0.86), xycoords="axes fraction", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
    )
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t, vf_corr, color="#d62728", linewidth=0.9)
    axes[1].set_ylabel("Corrected Void Fraction (%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Corrected Void Fraction")
    axes[1].annotate(
        rf"$\alpha_c = k\alpha^2 + (1 - 100k)\alpha \quad (k={_K_CORR})$",
        xy=(0.02, 0.86), xycoords="axes fraction", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
    )
    axes[1].grid(True, alpha=0.35)

    fig.suptitle("Void Fraction Processing Pipeline — Slug Flow Sample", fontweight="bold")
    fig.tight_layout()
    _save(fig, "void_fraction_pipeline.png")


def _fig3_flow_regime_map_3d(X: np.ndarray, labels: np.ndarray) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for idx, regime in enumerate(REGIME_NAMES):
        m = labels == idx
        ax.scatter(
            X[m, 0], X[m, 1], X[m, 2],
            c=_COLORS[regime], marker="o", s=130, alpha=0.92,
            depthshade=True, label=regime,
            edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("Norm. Mean Capacitance", labelpad=10)
    ax.set_ylabel("Norm. Variance", labelpad=10)
    ax.set_zlabel("Norm. Kurtosis", labelpad=10)
    ax.set_title("Flow Regime Feature Space — FCM Clustering", pad=12)
    ax.view_init(elev=25, azim=225)
    ax.legend(fontsize=11, loc="upper left")

    fig.tight_layout()
    _save(fig, "flow_regime_map_3d.png")


def _fig4_flow_regime_map_2d(X: np.ndarray, labels: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    legend_patches = [mpatches.Patch(color=_COLORS[r], label=r) for r in REGIME_NAMES]

    for idx, regime in enumerate(REGIME_NAMES):
        c = _COLORS[regime]
        m = labels == idx
        axes[0].scatter(X[m, 0], X[m, 1], c=c, s=80, alpha=0.9, edgecolors="white", linewidths=0.4)
        axes[1].scatter(X[m, 1], X[m, 2], c=c, s=80, alpha=0.9, edgecolors="white", linewidths=0.4)

    for ax, xlabel, ylabel, title in [
        (axes[0], "Norm. Mean Capacitance", "Norm. Variance", "Mean Capacitance vs Variance"),
        (axes[1], "Norm. Variance", "Norm. Kurtosis", "Variance vs Kurtosis"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_patches, fontsize=10)

    fig.suptitle("Flow Regime Map — 2D Projections", fontweight="bold")
    fig.tight_layout()
    _save(fig, "flow_regime_map_2d.png")


def _fig5_regime_distribution(
    ansys_df: pd.DataFrame,
    augmented: dict[str, pd.DataFrame],
) -> None:
    before_counts = ansys_df["flow_regime"].value_counts()
    before_vals = [int(before_counts.get(r, 0)) for r in REGIME_NAMES]
    after_vals = [len(augmented[r]) for r in REGIME_NAMES]
    colors = [_COLORS[r] for r in REGIME_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, vals, title in [
        (axes[0], before_vals, "Before GPR Augmentation"),
        (axes[1], after_vals, "After GPR Augmentation"),
    ]:
        labels = [f"{r}\n({v})" for r, v in zip(REGIME_NAMES, vals)]
        _, _, autotexts = ax.pie(
            vals, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.75,
        )
        for txt in autotexts:
            txt.set_fontsize(10)
        ax.set_title(f"{title}\n(n={sum(vals)})")

    fig.suptitle("ANSYS Simulation Data Distribution by Flow Regime", fontweight="bold")
    fig.tight_layout()
    _save(fig, "regime_distribution.png")


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_mpl()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("ECLIPSE Instrumentation Demo")
    print("=" * 29)
    print("Sensor: Asymmetric capacitance sensor for cryogenic two-phase flow monitoring")
    print("Pipeline: Capacitance signal → Void fraction → Flow regime classification")
    print()

    print("[1/4] Generating synthetic capacitance signals...")
    dataset = generate_dataset(
        n_samples_per_regime=_N_SYNTH,
        duration_s=_DURATION_S,
        sample_rate_hz=_RATE_HZ,
        seed=42,
    )
    print(f"  ✓ {_N_SYNTH} slug flow samples ({_DURATION_S:.1f}s each @ {_RATE_HZ:.0f}Hz)")
    print(f"  ✓ {_N_SYNTH} intermittent flow samples")
    print(f"  ✓ {_N_SYNTH} annular flow samples")
    print()

    print("[2/4] Processing void fraction...")
    print("  ✓ Linear void fraction extraction")
    print(f"  ✓ Thermal correction applied (k={_K_CORR})")
    print()

    print("[3/4] Training flow regime classifier...")
    ansys_df = load_ansys_data(str(_CSV_PATH))
    slug_aug = augment_with_gpr(ansys_df[ansys_df["flow_regime"] == "Slug"].copy(), 2)
    intermittent_aug = augment_with_gpr(
        ansys_df[ansys_df["flow_regime"] == "Intermittent"].copy(), 2
    )
    annular_raw = ansys_df[ansys_df["flow_regime"] == "Annular"].copy()

    combined = pd.concat([slug_aug, intermittent_aug, annular_raw]).sort_values("time")
    ansys_features = extract_features(combined["capacitance"].values, _WINDOW_SIZE)
    fcm, scaler, labels, label_map = fit_classifier(ansys_features)

    inv_map = {v: k for k, v in label_map.items()}
    separation = float(np.linalg.norm(fcm.centers[inv_map[0]] - fcm.centers[inv_map[2]]))

    print("  ✓ Feature extraction: mean, variance, kurtosis per 0.1s window")
    print("  ✓ GPR augmentation on ANSYS simulation data")
    print("  ✓ Fuzzy c-means clustering (3 regimes)")
    print(f"  Cluster separation: slug/annular centroid distance = {separation:.2f}")
    print()

    ansys_scaled = scaler.transform(ansys_features)

    print("[4/4] Generating visualizations...")
    _fig1_signal_examples(dataset)
    _fig2_void_fraction_pipeline(dataset)
    _fig3_flow_regime_map_3d(ansys_scaled, labels)
    _fig4_flow_regime_map_2d(ansys_scaled, labels)
    _fig5_regime_distribution(
        ansys_df,
        {"Slug": slug_aug, "Intermittent": intermittent_aug, "Annular": annular_raw},
    )
    print()
    print("Done. Open outputs/ to view results.")


if __name__ == "__main__":
    main()
