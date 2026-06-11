import numpy as np
import pytest

from instrumentation.void_fraction import (
    C_GAS_DEFAULT,
    C_LIQUID_DEFAULT,
    corrected_void_fraction,
    gas_permittivity,
    linear_void_fraction,
    thermal_correction_length,
)


def test_linear_void_fraction_bounds():
    assert linear_void_fraction(C_LIQUID_DEFAULT, C_LIQUID_DEFAULT, C_GAS_DEFAULT) == pytest.approx(0.0)
    assert linear_void_fraction(C_GAS_DEFAULT, C_LIQUID_DEFAULT, C_GAS_DEFAULT) == pytest.approx(100.0)


def test_linear_void_fraction_midpoint():
    C_mid = (C_LIQUID_DEFAULT + C_GAS_DEFAULT) / 2.0
    assert linear_void_fraction(C_mid, C_LIQUID_DEFAULT, C_GAS_DEFAULT) == pytest.approx(50.0)


def test_corrected_void_fraction_identity():
    alpha = np.linspace(0.0, 100.0, 50)
    np.testing.assert_allclose(corrected_void_fraction(alpha, k=0.0), alpha)


def test_thermal_correction_sign():
    assert thermal_correction_length(L0=1.0, alpha_cte=17.3e-6, delta_T=-200.0) < 0.0


def test_permittivity_physical_range():
    assert gas_permittivity(A_g=4.3e-4, P=101325.0, T=90.0) > 1.0
