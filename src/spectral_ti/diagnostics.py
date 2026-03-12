"""
Constitutive diagnostics: eigenvalue trajectories, admissibility scans,
and the density-field benchmark used by Figure 4 panels (b, c).
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from .constitutive import isotropic_voigt, isotropic_voigt_stiffness, MaterialParams, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Coupled normal-block eigenvalue trajectory  (Figure 3 panel a)
# ---------------------------------------------------------------------------
def coupled_normal_block_min_eigs(
    rhos: np.ndarray,
    p1: float,
    p3: float,
    pc: float,
    E: float = 1.0,
    nu: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum eigenvalue of the 2×2 coupled normal-mode block for the baseline
    ICP law versus the spectral (geometric-mean) coupling.

    Returns (min_eig_icp, min_eig_spectral) arrays of the same shape as *rhos*.
    """
    c11, c12, _ = isotropic_voigt_stiffness(E=E, nu=nu)
    k1_0 = c11
    k2_0 = c11 + c12
    k_coup_0 = np.sqrt(2.0) * c12

    min_eig_icp = []
    min_eig_spec = []
    p_coup_spec = 0.5 * (p1 + p3)

    for rho in rhos:
        C_icp = np.array([
            [rho**p1 * k2_0,          rho**pc * k_coup_0],
            [rho**pc * k_coup_0,      rho**p3 * k1_0],
        ])
        C_spec = np.array([
            [rho**p1 * k2_0,               rho**p_coup_spec * k_coup_0],
            [rho**p_coup_spec * k_coup_0,  rho**p3 * k1_0],
        ])
        min_eig_icp.append(np.min(np.linalg.eigvalsh(C_icp)))
        min_eig_spec.append(np.min(np.linalg.eigvalsh(C_spec)))

    return np.array(min_eig_icp), np.array(min_eig_spec)


# ---------------------------------------------------------------------------
# Directional Young-modulus surface  (Figure 3 panels b, c)
# ---------------------------------------------------------------------------
def compute_directional_young_surface(
    C_voigt: np.ndarray,
    THETA: np.ndarray,
    PHI: np.ndarray,
    cap: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Directional Young modulus E(n) = 1 / (v^T S v) on a (theta, phi) grid.

    Returns (X, Y, Z, E_dir) Cartesian coordinates and raw modulus values.
    """
    n1 = np.sin(PHI) * np.cos(THETA)
    n2 = np.sin(PHI) * np.sin(THETA)
    n3 = np.cos(PHI)

    v = np.array([n1**2, n2**2, n3**2, n2 * n3, n1 * n3, n1 * n2])

    S = np.linalg.inv(C_voigt)
    Sv = np.einsum('ij,jkl->ikl', S, v)
    compliance = np.sum(v * Sv, axis=0)

    E_dir = np.full_like(compliance, np.nan, dtype=float)
    positive = np.isfinite(compliance) & (compliance > 0.0)
    E_dir[positive] = 1.0 / compliance[positive]
    E_dir = np.clip(E_dir, 0.0, cap)

    X = E_dir * n1
    Y = E_dir * n2
    Z = E_dir * n3
    return X, Y, Z, E_dir


# ---------------------------------------------------------------------------
# Density-field benchmark  (Figure 4 panels b, c)
# ---------------------------------------------------------------------------
@dataclass
class DensityBenchmarkParams:
    rho_crit: float
    center: float
    amplitude: float
    rho_min: float
    rho_max: float


def baseline_critical_density(
    C0: np.ndarray,
    params: MaterialParams = DEFAULT_PARAMS,
) -> float:
    """
    Critical density at which the baseline ICP coupled normal-mode determinant
    first becomes non-positive.
    """
    delta = params.p_t + params.p_l - 2.0 * params.p_c
    if delta <= 0.0:
        raise ValueError("Need p_t + p_l - 2 p_c > 0 for a finite critical density.")

    k1 = C0[2, 2]
    k2 = C0[0, 0] + C0[0, 1]
    k_coup = np.sqrt(2.0) * C0[0, 2]

    ratio = (k_coup * k_coup) / (k1 * k2)
    rho_crit = ratio ** (1.0 / delta)
    return float(np.clip(rho_crit, 1e-6, 1.0))


def benchmark_density_parameters(
    C0: np.ndarray,
    params: MaterialParams = DEFAULT_PARAMS,
) -> DensityBenchmarkParams:
    """Design the density field for the forward-problem benchmark."""
    rho_crit = baseline_critical_density(C0, params)
    center = min(0.92, rho_crit + 0.06)
    amplitude = 0.12
    rho_min = max(0.28, center - 1.25 * amplitude)
    rho_max = min(0.88, center + 1.25 * amplitude)
    return DensityBenchmarkParams(
        rho_crit=rho_crit, center=center, amplitude=amplitude,
        rho_min=rho_min, rho_max=rho_max,
    )


def density_field(xc, yc, zc, params: DensityBenchmarkParams) -> float:
    """Smooth non-uniform density field centred near rho_crit."""
    rho = (
        params.center
        + params.amplitude
        * np.sin(2.0 * np.pi * xc)
        * np.sin(2.0 * np.pi * yc)
        * np.sin(2.0 * np.pi * zc)
    )
    return float(np.clip(rho, params.rho_min, params.rho_max))


def constitutive_admissibility_scan(
    N: int,
    constitutive: Callable,
    C0: np.ndarray,
    density_params: DensityBenchmarkParams,
    neg_tol: float = -1e-12,
) -> Tuple[float, float]:
    """
    Scan over all elements: returns (inadmissible_fraction, min_local_lambda).
    """
    h = 1.0 / N
    total = N * N * N
    neg_count = 0
    min_lam = np.inf

    for ez in range(N):
        zc = (ez + 0.5) * h
        for ey in range(N):
            yc = (ey + 0.5) * h
            for ex in range(N):
                xc = (ex + 0.5) * h
                rho = density_field(xc, yc, zc, density_params)
                C = constitutive(rho, C0)
                lam_min = float(np.min(np.linalg.eigvalsh(C)))
                min_lam = min(min_lam, lam_min)
                if lam_min < neg_tol:
                    neg_count += 1

    return neg_count / total, min_lam
