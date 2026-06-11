from __future__ import annotations

import numpy as np

C_LIQUID_DEFAULT: float = 1.113e-10 * 1.51 * (10 * 10) / 10
C_GAS_DEFAULT: float = 1.113e-10 * 1.000494 * (10 * 10) / 10


def linear_void_fraction(
    C_measured: float | np.ndarray,
    C_liquid: float,
    C_gas: float,
) -> float | np.ndarray:
    """
    α_linear = (C_L - C_M) / (C_L - C_G) × 100  [%]

    C_liquid, C_gas, C_measured in Farads.
    Returns void fraction in [0, 100] percent.
    """
    return (C_liquid - C_measured) / (C_liquid - C_gas) * 100.0


def corrected_void_fraction(
    alpha_linear: float | np.ndarray,
    k: float,
) -> float | np.ndarray:
    """
    α_corrected = k·α² + (1 − 100k)·α  [%]

    alpha_linear in percent. k proportional to electrode separation distance
    (0.001–0.005 for 10-inch pipe, calibrated via FEM).
    """
    return k * alpha_linear**2 + (1.0 - 100.0 * k) * alpha_linear


def thermal_correction_length(
    L0: float,
    alpha_cte: float,
    delta_T: float,
) -> float:
    """
    ΔL = L₀ × α_CTE × ΔT  [m]

    L0 in m, alpha_cte in m/m/K, delta_T in K (negative for cooling).
    """
    return L0 * alpha_cte * delta_T


def gas_permittivity(
    A_g: float,
    P: float,
    T: float,
) -> float:
    """
    ε_g = 1 + A_g × (P/T)

    P in Pa, T in K. Returns dimensionless relative permittivity (> 1).
    """
    return 1.0 + A_g * (P / T)


def liquid_permittivity(
    A_l: float,
    B_l: float,
    T: float,
) -> float:
    """
    ε_l = A_l + B_l / T

    T in K, B_l in K. Returns dimensionless relative permittivity.
    """
    return A_l + B_l / T


def process_signal(
    capacitance: np.ndarray,
    C_liquid: float,
    C_gas: float,
    k: float = 0.002,
) -> np.ndarray:
    """
    Full pipeline: raw capacitance (F) → corrected void fraction (%).

    Applies linear_void_fraction then corrected_void_fraction with calibration constant k.
    """
    alpha = linear_void_fraction(capacitance, C_liquid, C_gas)
    return corrected_void_fraction(alpha, k)
