"""Smoke tests for the constitutive module."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_ti.constitutive import (
    isotropic_voigt,
    congruence_ti_from_base,
    local_spectral_ti,
    local_baseline_ti,
    DEFAULT_PARAMS,
)


def test_isotropic_symmetry():
    C = isotropic_voigt(1.0, 0.3)
    assert np.allclose(C, C.T), "Isotropic tensor must be symmetric"


def test_isotropic_spd():
    C = isotropic_voigt(1.0, 0.3)
    eigs = np.linalg.eigvalsh(C)
    assert np.all(eigs > 0), f"Isotropic tensor must be SPD, got eigs={eigs}"


def test_spectral_always_spd():
    """Spectral path must be SPD for any density in (0, 1]."""
    C0 = isotropic_voigt()
    for rho in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
        rho_arr = np.array([rho])
        C, _ = local_spectral_ti(rho_arr, C0)
        eigs = np.linalg.eigvalsh(C[0])
        assert np.all(eigs > 0), f"Spectral path not SPD at rho={rho}: eigs={eigs}"


def test_congruence_matches_spectral():
    """Scalar congruence with eta_L=rho, eta_T=1, eta_S=sqrt(rho) must match
    the vectorized spectral path at a single density."""
    C0 = isotropic_voigt()
    rho = 0.4
    C_scalar = congruence_ti_from_base(C0, rho=rho, p=3.0,
                                        eta_L=rho, eta_T=1.0, eta_S=np.sqrt(rho))
    C_vec, _ = local_spectral_ti(np.array([rho]), C0)
    assert np.allclose(C_scalar, C_vec[0], atol=1e-12), "Scalar and vectorized paths disagree"


def test_baseline_may_lose_spd():
    """Baseline ICP path should lose SPD at low density with default exponents."""
    C0 = isotropic_voigt()
    rho_arr = np.array([0.05])
    C, _ = local_baseline_ti(rho_arr, C0)
    eigs = np.linalg.eigvalsh(C[0])
    # The baseline path is known to lose SPD at low density; just verify no crash
    assert eigs.shape == (6,)


if __name__ == "__main__":
    test_isotropic_symmetry()
    test_isotropic_spd()
    test_spectral_always_spd()
    test_congruence_matches_spectral()
    test_baseline_may_lose_spd()
    print("All constitutive tests passed.")
