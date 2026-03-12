"""
Constitutive interpolation laws for TI topology optimization.

This module provides the unified material model used across all figures and
tables.  Two interpolation families are implemented:

  1. **Spectral / congruence path** — algebraically guaranteed positive-definite
     for every density in (0, 1].
  2. **Baseline component-wise (ICP) path** — the conventional approach whose
     coupled normal block may lose positive-definiteness at low density.

Engineering Voigt ordering: 11, 22, 33, 23, 13, 12.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Material parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialParams:
    """Default material / penalization parameters shared across the paper."""
    E0: float = 1.0
    nu0: float = 0.30
    rho_floor: float = 1e-3
    # Baseline component-wise exponents
    p_t: float = 3.0    # transverse / in-plane normal
    p_l: float = 5.0    # longitudinal normal (build direction)
    p_c: float = 3.0    # cross-coupling C13, C23
    p_s: float = 4.0    # out-of-plane shear C44, C55
    # Spectral outer exponent
    p_outer: float = 3.0


DEFAULT_PARAMS = MaterialParams()


# ---------------------------------------------------------------------------
# Isotropic base stiffness
# ---------------------------------------------------------------------------
def isotropic_voigt(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """6×6 isotropic stiffness tensor in engineering Voigt ordering."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6), dtype=float)
    C[0:3, 0:3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def isotropic_voigt_stiffness(E: float = 1.0, nu: float = 0.3):
    """Return (c11, c12, mu) for compact constitutive expressions."""
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c11 = fac * (1.0 - nu)
    c12 = fac * nu
    mu = E / (2.0 * (1.0 + nu))
    return c11, c12, mu


# ---------------------------------------------------------------------------
# Scalar congruence law  (single-element; used by Table 1 / Figure 4 panel a)
# ---------------------------------------------------------------------------
def congruence_ti_from_base(
    C0: np.ndarray,
    rho: float,
    p: float,
    eta_L: float,
    eta_T: float,
    eta_S: float,
) -> np.ndarray:
    """
    Engineering-Voigt TI tensor via the congruence law (scalar version).

        C_ij^eff = S * eta_i * eta_j * C_ij^0

    with S = rho^p.
    """
    S = rho ** p
    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = S * eta_T**2 * C0[0, 0]
    C[1, 1] = S * eta_T**2 * C0[1, 1]
    C[0, 1] = C[1, 0] = S * eta_T**2 * C0[0, 1]

    C[0, 2] = C[2, 0] = S * eta_L * eta_T * C0[0, 2]
    C[1, 2] = C[2, 1] = S * eta_L * eta_T * C0[1, 2]
    C[2, 2] = S * eta_L**2 * C0[2, 2]

    C[3, 3] = S * eta_S**2 * C0[3, 3]
    C[4, 4] = S * eta_S**2 * C0[4, 4]
    C[5, 5] = S * eta_T**2 * C0[5, 5]
    return C


# ---------------------------------------------------------------------------
# Vectorized constitutive laws  (used by Figure 5 / Figure 6 / Table 2)
# ---------------------------------------------------------------------------
def local_spectral_ti(
    rho_phys: np.ndarray,
    C0: np.ndarray,
    params: MaterialParams = DEFAULT_PARAMS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized spectral / congruence path: returns (C, dC/drho).

    Exponent structure:
        C11, C22, C12, C66  ~ rho^3
        C13, C23, C44, C55  ~ rho^4
        C33                 ~ rho^5
    """
    rb = np.maximum(rho_phys, params.rho_floor)

    c11 = C0[0, 0]; c12 = C0[0, 1]; c13 = C0[0, 2]
    c33 = C0[2, 2]; c44 = C0[3, 3]; c66 = C0[5, 5]

    C = np.zeros((rho_phys.size, 6, 6), dtype=float)
    dC = np.zeros_like(C)

    r3 = rb**3; r4 = rb**4; r5 = rb**5

    C[:, 0, 0] = C[:, 1, 1] = r3 * c11
    C[:, 0, 1] = C[:, 1, 0] = r3 * c12
    C[:, 0, 2] = C[:, 2, 0] = r4 * c13
    C[:, 1, 2] = C[:, 2, 1] = r4 * c13
    C[:, 2, 2] = r5 * c33
    C[:, 3, 3] = C[:, 4, 4] = r4 * c44
    C[:, 5, 5] = r3 * c66

    dC[:, 0, 0] = dC[:, 1, 1] = 3.0 * rb**2 * c11
    dC[:, 0, 1] = dC[:, 1, 0] = 3.0 * rb**2 * c12
    dC[:, 0, 2] = dC[:, 2, 0] = 4.0 * rb**3 * c13
    dC[:, 1, 2] = dC[:, 2, 1] = 4.0 * rb**3 * c13
    dC[:, 2, 2] = 5.0 * rb**4 * c33
    dC[:, 3, 3] = dC[:, 4, 4] = 4.0 * rb**3 * c44
    dC[:, 5, 5] = 3.0 * rb**2 * c66
    return C, dC


def local_baseline_ti(
    rho_phys: np.ndarray,
    C0: np.ndarray,
    params: MaterialParams = DEFAULT_PARAMS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized baseline component-wise (ICP) path: returns (C, dC/drho).

    Exponents: p_t for in-plane normal, p_l for C33, p_c for C13/C23, p_s for shear.
    """
    rb = np.maximum(rho_phys, params.rho_floor)
    p_t, p_l, p_c, p_s = params.p_t, params.p_l, params.p_c, params.p_s

    c11 = C0[0, 0]; c12 = C0[0, 1]; c13 = C0[0, 2]
    c33 = C0[2, 2]; c44 = C0[3, 3]; c66 = C0[5, 5]

    C = np.zeros((rho_phys.size, 6, 6), dtype=float)
    dC = np.zeros_like(C)

    rt = rb**p_t; rc = rb**p_c; rl = rb**p_l; rs = rb**p_s

    C[:, 0, 0] = C[:, 1, 1] = rt * c11
    C[:, 0, 1] = C[:, 1, 0] = rt * c12
    C[:, 0, 2] = C[:, 2, 0] = rc * c13
    C[:, 1, 2] = C[:, 2, 1] = rc * c13
    C[:, 2, 2] = rl * c33
    C[:, 3, 3] = C[:, 4, 4] = rs * c44
    C[:, 5, 5] = rt * c66

    dC[:, 0, 0] = dC[:, 1, 1] = p_t * rb**(p_t - 1.0) * c11
    dC[:, 0, 1] = dC[:, 1, 0] = p_t * rb**(p_t - 1.0) * c12
    dC[:, 0, 2] = dC[:, 2, 0] = p_c * rb**(p_c - 1.0) * c13
    dC[:, 1, 2] = dC[:, 2, 1] = p_c * rb**(p_c - 1.0) * c13
    dC[:, 2, 2] = p_l * rb**(p_l - 1.0) * c33
    dC[:, 3, 3] = dC[:, 4, 4] = p_s * rb**(p_s - 1.0) * c44
    dC[:, 5, 5] = p_t * rb**(p_t - 1.0) * c66
    return C, dC


def constitutive_builder(method: str, params: MaterialParams = DEFAULT_PARAMS):
    """Return the appropriate vectorized constitutive law."""
    if method == "spectral":
        return lambda rho, C0: local_spectral_ti(rho, C0, params)
    if method == "icp":
        return lambda rho, C0: local_baseline_ti(rho, C0, params)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Scalar baseline law  (used by Figure 4 panels b, c)
# ---------------------------------------------------------------------------
def baseline_componentwise_tensor(
    rho: float,
    C0: np.ndarray,
    params: MaterialParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Single-element baseline component-wise penalization."""
    p_t, p_l, p_c, p_s = params.p_t, params.p_l, params.p_c, params.p_s
    C = np.zeros((6, 6), dtype=float)
    C[0, 0] = rho**p_t * C0[0, 0]
    C[1, 1] = rho**p_t * C0[1, 1]
    C[0, 1] = C[1, 0] = rho**p_t * C0[0, 1]
    C[0, 2] = C[2, 0] = rho**p_c * C0[0, 2]
    C[1, 2] = C[2, 1] = rho**p_c * C0[1, 2]
    C[2, 2] = rho**p_l * C0[2, 2]
    C[3, 3] = rho**p_s * C0[3, 3]
    C[4, 4] = rho**p_s * C0[4, 4]
    C[5, 5] = rho**p_t * C0[5, 5]
    return C


def spectral_congruence_tensor(
    rho: float,
    C0: np.ndarray,
    p: float = 3.0,
) -> np.ndarray:
    """
    Single-element congruence-based admissible path:
        S = rho^p, eta_T = 1, eta_L = rho, eta_S = sqrt(rho).
    """
    return congruence_ti_from_base(
        C0=C0, rho=rho, p=p,
        eta_L=rho, eta_T=1.0, eta_S=np.sqrt(rho),
    )


# ---------------------------------------------------------------------------
# TI tensor from explicit exponents  (used by Figure 3)
# ---------------------------------------------------------------------------
def build_ti_voigt_from_path(
    rho: float,
    p1: float,
    p3: float,
    p13: float,
    p44: float,
    E: float = 1.0,
    nu: float = 0.3,
) -> np.ndarray:
    """
    Minimal TI constitutive family for the eigenvalue-trajectory analysis.

    - In-plane normal block: exponent p1
    - Build-direction normal stiffness: exponent p3
    - Cross-coupling C13=C23: exponent p13
    - Out-of-plane shear C44=C55: exponent p44
    - In-plane shear C66 = (C11 - C12)/2  (TI constraint)
    """
    c11, c12, mu = isotropic_voigt_stiffness(E=E, nu=nu)

    C11 = rho**p1 * c11
    C12 = rho**p1 * c12
    C33 = rho**p3 * c11
    C13 = rho**p13 * c12
    C44 = rho**p44 * mu
    C66 = 0.5 * (C11 - C12)

    C = np.array([
        [C11, C12, C13, 0.0, 0.0, 0.0],
        [C12, C11, C13, 0.0, 0.0, 0.0],
        [C13, C13, C33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, C66],
    ], dtype=float)
    return C


# ---------------------------------------------------------------------------
# Directional modulus analysis
# ---------------------------------------------------------------------------
def directional_young_modulus(C: np.ndarray, n: np.ndarray) -> float:
    """Directional Young modulus E(n) = 1 / (v^T S v)."""
    S = np.linalg.inv(C)
    n1, n2, n3 = n
    v = np.array([n1 * n1, n2 * n2, n3 * n3, n2 * n3, n1 * n3, n1 * n2])
    denom = float(v @ S @ v)
    if denom <= 0.0:
        return np.nan
    return 1.0 / denom


def anisotropy_ratio(C: np.ndarray, n_theta: int = 240, n_phi: int = 121) -> float:
    """E_max / E_min over the unit sphere."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phis = np.linspace(0.0, np.pi, n_phi)
    vals = []
    for phi in phis:
        sp_ = np.sin(phi)
        cp_ = np.cos(phi)
        for theta in thetas:
            n = np.array([sp_ * np.cos(theta), sp_ * np.sin(theta), cp_])
            vals.append(directional_young_modulus(C, n))
    vals = np.array(vals)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    return float(vals.max() / vals.min())


def calibrate_eta_family_for_xi(
    target_xi: float,
    C0: np.ndarray,
    rho0: float,
    p: float,
    tol: float = 1e-4,
) -> Tuple[float, float, float, float]:
    """
    Controlled anisotropy family for MMS study:
        eta_T = 1, eta_L = alpha, eta_S = sqrt(alpha).
    Calibrates alpha by bisection so that E_max/E_min matches target_xi.
    """
    if abs(target_xi - 1.0) < 1e-14:
        alpha = 1.0
        C = congruence_ti_from_base(C0, rho=rho0, p=p,
                                     eta_L=alpha, eta_T=1.0, eta_S=np.sqrt(alpha))
        return alpha, 1.0, np.sqrt(alpha), anisotropy_ratio(C)

    lo, hi = 1e-4, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        C_mid = congruence_ti_from_base(C0, rho=rho0, p=p,
                                         eta_L=mid, eta_T=1.0, eta_S=np.sqrt(mid))
        xi_mid = anisotropy_ratio(C_mid, n_theta=120, n_phi=61)
        if xi_mid > target_xi:
            lo = mid
        else:
            hi = mid
        if abs(xi_mid - target_xi) / target_xi < tol:
            break
    alpha = 0.5 * (lo + hi)
    C = congruence_ti_from_base(C0, rho=rho0, p=p,
                                 eta_L=alpha, eta_T=1.0, eta_S=np.sqrt(alpha))
    xi_eff = anisotropy_ratio(C)
    return alpha, 1.0, np.sqrt(alpha), xi_eff
