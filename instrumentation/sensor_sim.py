from __future__ import annotations

import numpy as np
from numpy.random import Generator

from instrumentation.void_fraction import C_GAS_DEFAULT, C_LIQUID_DEFAULT

_REGIMES = frozenset({"slug", "intermittent", "annular"})


def _slug_vf(t: np.ndarray, rng: Generator, duration_s: float) -> np.ndarray:
    n = len(t)
    vf = np.full(n, 0.3)

    n_spikes = max(2, int(rng.uniform(2.0, 5.0) * duration_s))
    centers = rng.uniform(0.0, duration_s, n_spikes)
    heights = rng.uniform(0.4, 0.6, n_spikes)

    for center, height in zip(centers, heights):
        vf += height * np.exp(-((t - center) / 0.02) ** 2)

    vf += rng.normal(0.0, 0.02, n)
    return np.clip(vf, 0.0, 1.0)


def _intermittent_vf(t: np.ndarray, rng: Generator, duration_s: float) -> np.ndarray:
    n = len(t)
    vf = np.full(n, 0.5)
    vf += 0.15 * np.sin(2.0 * np.pi * rng.uniform(3.0, 8.0) * t + rng.uniform(0.0, 2.0 * np.pi))

    n_spikes = max(1, int(rng.uniform(1.0, 2.0) * duration_s))
    centers = rng.uniform(0.0, duration_s, n_spikes)
    heights = rng.uniform(0.1, 0.2, n_spikes)

    for center, height in zip(centers, heights):
        vf += height * np.exp(-((t - center) / 0.04) ** 2)

    vf += rng.normal(0.0, 0.015, n)
    return np.clip(vf, 0.0, 1.0)


def _annular_vf(t: np.ndarray, rng: Generator) -> np.ndarray:
    n = len(t)
    vf = np.full(n, rng.uniform(0.85, 0.95))
    vf += 0.03 * np.sin(2.0 * np.pi * rng.uniform(0.5, 1.0) * t + rng.uniform(0.0, 2.0 * np.pi))
    vf += rng.normal(0.0, 0.005, n)
    return np.clip(vf, 0.0, 1.0)


def generate_signal(
    regime: str,
    duration_s: float,
    sample_rate_hz: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic capacitance signal for a cryogenic two-phase flow regime.

    regime: 'slug', 'intermittent', or 'annular'
    duration_s: signal duration in seconds
    sample_rate_hz: sampling frequency in Hz (>= 200 per sensor spec)
    seed: integer seed for reproducibility; None gives a non-deterministic signal

    Returns (time, capacitance) arrays of length int(duration_s * sample_rate_hz).
    Mean capacitance ordering by construction: slug > intermittent > annular.
    Variance ordering by construction: slug > intermittent > annular.
    """
    if regime not in _REGIMES:
        raise ValueError(f"regime must be one of {sorted(_REGIMES)}, got {regime!r}")
    if sample_rate_hz < 200:
        raise ValueError(f"sample_rate_hz must be >= 200 Hz, got {sample_rate_hz}")

    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sample_rate_hz)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)

    if regime == "slug":
        vf = _slug_vf(t, rng, duration_s)
    elif regime == "intermittent":
        vf = _intermittent_vf(t, rng, duration_s)
    else:
        vf = _annular_vf(t, rng)

    return t, C_LIQUID_DEFAULT - vf * (C_LIQUID_DEFAULT - C_GAS_DEFAULT)


def generate_dataset(
    n_samples_per_regime: int = 200,
    duration_s: float = 2.0,
    sample_rate_hz: float = 200.0,
    seed: int = 42,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Generate a labeled dataset of synthetic capacitance signals for all three flow regimes.

    Returns dict with keys 'slug', 'intermittent', 'annular', each containing a list of
    n_samples_per_regime (time, capacitance) tuples. Fully reproducible with fixed seed.
    """
    rng = np.random.default_rng(seed)
    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for regime in ("slug", "intermittent", "annular"):
        samples = []
        for _ in range(n_samples_per_regime):
            child_seed = int(rng.integers(0, 2**31))
            samples.append(generate_signal(regime, duration_s, sample_rate_hz, seed=child_seed))
        result[regime] = samples

    return result
