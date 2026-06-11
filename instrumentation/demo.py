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
from sklearn.preprocessing import StandardScaler

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
_N_SCATTER: int = 20

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


def _log_transform(features: np.ndarray) -> np.ndarray:
    result = features.copy()
    result[:, 1] = np.log10(np.maximum(features[:, 1], 1e-60))
    result[:, 2] = np.log10(np.maximum(features[:, 2], 1e-60))
    return result


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


def _fig3_flow_regime_map_3d(
    ansys_viz: np.ndarray,
    ansys_y: np.ndarray,
    synth_viz: np.ndarray,
    synth_y: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for idx, regime in enumerate(REGIME_NAMES):
        c = _COLORS[regime]
        ma, ms = ansys_y == idx, synth_y == idx
        ax.scatter(
            ansys_viz[ma, 0], ansys_viz[ma, 1], ansys_viz[ma, 2],
            c=c, marker="o", s=70, alpha=0.9, depthshade=True,
            label=f"{regime} (ANSYS)",
        )
        ax.scatter(
            synth_viz[ms, 0], synth_viz[ms, 1], synth_viz[ms, 2],
            c=c, marker="^", s=14, alpha=0.25, depthshade=True,
            label=f"{regime} (Synthetic)",
        )

    ax.set_xlabel("Norm. Mean Capacitance", labelpad=10)
    ax.set_ylabel("Norm. log Variance", labelpad=10)
    ax.set_zlabel("Norm. log Kurtosis", labelpad=10)
    ax.set_title("Flow Regime Map — 3D Feature Space", pad=14)
    ax.view_init(elev=25, azim=45)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    _save(fig, "flow_regime_map_3d.png")


def _fig4_flow_regime_map_2d(
    ansys_viz: np.ndarray,
    ansys_y: np.ndarray,
    synth_viz: np.ndarray,
    synth_y: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    legend_patches = [mpatches.Patch(color=_COLORS[r], label=r) for r in REGIME_NAMES]

    for idx in range(len(REGIME_NAMES)):
        c = _COLORS[REGIME_NAMES[idx]]
        ma, ms = ansys_y == idx, synth_y == idx

        axes[0].scatter(ansys_viz[ma, 0], ansys_viz[ma, 1], c=c, marker="o", s=45, alpha=0.9)
        axes[0].scatter(synth_viz[ms, 0], synth_viz[ms, 1], c=c, marker="^", s=10, alpha=0.25)

        axes[1].scatter(ansys_viz[ma, 1], ansys_viz[ma, 2], c=c, marker="o", s=45, alpha=0.9)
        axes[1].scatter(synth_viz[ms, 1], synth_viz[ms, 2], c=c, marker="^", s=10, alpha=0.25)

    for ax, xlabel, ylabel, title in [
        (axes[0], "Norm. Mean Capacitance", "Norm. log Variance", "Mean Capacitance vs Variance"),
        (axes[1], "Norm. log Variance", "Norm. log Kurtosis", "Variance vs Kurtosis"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_patches, fontsize=10)
        ax.annotate(
            "● ANSYS  ▲ Synthetic",
            xy=(0.99, 0.02), xycoords="axes fraction",
            ha="right", fontsize=9, color="gray",
        )

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
    fcm, _, _, label_map = fit_classifier(ansys_features)

    regime_row_labels = np.concatenate([
        np.zeros(len(slug_aug), dtype=int),
        np.ones(len(intermittent_aug), dtype=int),
        np.full(len(annular_raw), 2, dtype=int),
    ])
    ansys_true_labels = np.array([
        np.bincount(
            regime_row_labels[i * _WINDOW_SIZE : (i + 1) * _WINDOW_SIZE],
            minlength=3,
        ).argmax()
        for i in range(len(ansys_features))
    ])

    inv_map = {v: k for k, v in label_map.items()}
    separation = float(np.linalg.norm(fcm.centers[inv_map[0]] - fcm.centers[inv_map[2]]))

    print("  ✓ Feature extraction: mean, variance, kurtosis per 0.1s window")
    print("  ✓ GPR augmentation on ANSYS simulation data")
    print("  ✓ Fuzzy c-means clustering (3 regimes)")
    print(f"  Cluster separation: slug/annular centroid distance = {separation:.2f}")
    print()

    synth_feats_list: list[np.ndarray] = []
    synth_label_list: list[int] = []
    for true_idx, key in enumerate(_REGIME_KEYS):
        for _, C in dataset[key][: _N_SCATTER]:
            f = extract_features(C, _WINDOW_SIZE)
            synth_feats_list.append(f)
            synth_label_list.extend([true_idx] * len(f))
    synth_feats_raw = np.vstack(synth_feats_list)
    synth_labels = np.array(synth_label_list)

    ansys_log = _log_transform(ansys_features)
    synth_log = _log_transform(synth_feats_raw)
    viz_scaler = StandardScaler().fit(np.vstack([ansys_log, synth_log]))
    ansys_viz = viz_scaler.transform(ansys_log)
    synth_viz = viz_scaler.transform(synth_log)

    print("[4/4] Generating visualizations...")
    _fig1_signal_examples(dataset)
    _fig2_void_fraction_pipeline(dataset)
    _fig3_flow_regime_map_3d(ansys_viz, ansys_true_labels, synth_viz, synth_labels)
    _fig4_flow_regime_map_2d(ansys_viz, ansys_true_labels, synth_viz, synth_labels)
    _fig5_regime_distribution(
        ansys_df,
        {"Slug": slug_aug, "Intermittent": intermittent_aug, "Annular": annular_raw},
    )
    print()
    print("Done. Open outputs/ to view results.")


if __name__ == "__main__":
    main()
