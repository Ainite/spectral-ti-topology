"""
Topology optimization driver with OC (Optimality Criteria) update.

This module unifies the optimization loops from Figure 5 (cantilever),
Figure 6 (MBB beam), and Table 2 (cantilever quantitative data) into a
single parameterized driver.  Differences between the three are captured
by :class:`OptConfig` and whether the FE cache has passive regions.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve

from .constitutive import isotropic_voigt, constitutive_builder, MaterialParams, DEFAULT_PARAMS
from .mandel import rotation_about_y, mandel_rotation_matrix_from_R, rotate_local_family_to_global
from .fem import assemble_global_stiffness_from_basis, FECacheMBB


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OptConfig:
    """All knobs that differ between cantilever / MBB / Table 2 runs."""
    volfrac: float = 0.17
    max_iter: int = 36
    filter_sigma: float = 1.10
    move: float = 0.08
    eta_proj: float = 0.45
    # Beta continuation schedule
    beta_final: float = 5.5
    beta_milestones: Tuple[float, ...] = (0.25, 0.50, 0.75)
    beta_values: Tuple[float, ...] = (1.0, 2.0, 3.5)
    # Material
    params: MaterialParams = field(default_factory=lambda: DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Filter / projection / beta
# ---------------------------------------------------------------------------
def apply_filter(rho_grid: np.ndarray, sigma: float) -> np.ndarray:
    """Density filter (symmetric Gaussian convolution)."""
    return gaussian_filter(rho_grid, sigma=sigma, mode='reflect')


def heaviside_projection(rho_bar: np.ndarray, beta: float, eta: float = 0.45):
    """Smooth Heaviside projection.  Returns (rho_phys, drho/drho_bar)."""
    num = np.tanh(beta * eta) + np.tanh(beta * (rho_bar - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    rho_phys = num / den
    drho = beta * (1.0 - np.tanh(beta * (rho_bar - eta))**2) / den
    return rho_phys, drho


def beta_schedule(it: int, max_iter: int, config: OptConfig) -> float:
    """Piecewise-constant beta continuation."""
    s = it / max(1, max_iter - 1)
    # Walk milestones in reverse; return the first match
    for milestone, val in zip(
        reversed(config.beta_milestones), reversed(config.beta_values)
    ):
        if s >= milestone:
            return config.beta_final
    # Before the first milestone
    return config.beta_values[0] if config.beta_values else 1.0


def _beta_schedule_impl(it, max_iter, config):
    """Faithful re-implementation of the stepped schedule."""
    s = it / max(1, max_iter - 1)
    milestones = list(config.beta_milestones) + [1.0]
    values = list(config.beta_values) + [config.beta_final]
    for i in range(len(milestones) - 1, -1, -1):
        if s >= milestones[i]:
            return values[i]
    return values[0]


# ---------------------------------------------------------------------------
# Passive-region helpers
# ---------------------------------------------------------------------------
def _has_passive(cache) -> bool:
    return isinstance(cache, FECacheMBB) and cache.passive_solid_mask is not None


def enforce_passive_regions(rho_vec, cache, solid_value=1.0):
    """Set passive solid elements to *solid_value* (no-op for cantilever)."""
    if not _has_passive(cache):
        return rho_vec
    out = rho_vec.copy()
    out[cache.passive_solid_mask.ravel()] = solid_value
    return out


# ---------------------------------------------------------------------------
# Core optimization loop
# ---------------------------------------------------------------------------
@dataclass
class OptResult:
    """Return value of :func:`run_optimization`."""
    rho_phys: np.ndarray         # (nx, ny, nz) final physical density
    rho_raw: np.ndarray          # (nel,) final design variable
    final_compliance: float
    final_volume: float
    final_min_eig: float
    final_inad_frac: float
    worst_local_eig: float
    max_inad_frac: float
    max_pos_sens_frac: float


def run_optimization(
    theta_deg: float,
    method: str,
    cache,
    config: OptConfig = OptConfig(),
    verbose: bool = True,
) -> OptResult:
    """
    OC-based topology optimization with density filtering and Heaviside
    projection.

    Works for both cantilever (:class:`FECache`) and MBB (:class:`FECacheMBB`)
    meshes — passive regions are handled transparently.
    """
    C0 = isotropic_voigt(config.params.E0, config.params.nu0)
    R = rotation_about_y(theta_deg)
    Q_m = mandel_rotation_matrix_from_R(R)
    local_builder = constitutive_builder(method, config.params)
    rho_floor = config.params.rho_floor

    has_passive = _has_passive(cache)

    rho = np.full(cache.nel, config.volfrac, dtype=float)
    if has_passive:
        rho = enforce_passive_regions(rho, cache, solid_value=1.0)

    worst_local_eig = np.inf
    max_inad_frac = 0.0
    max_pos_sens_frac = 0.0

    for it in range(config.max_iter):
        if verbose:
            print(
                f"    [{method.upper():<8}] theta={theta_deg:>3}° | "
                f"iter {it + 1:>3}/{config.max_iter}",
                end="\r",
            )

        beta = _beta_schedule_impl(it, config.max_iter, config)

        # --- filter + project ---
        rho_bar = apply_filter(
            rho.reshape((cache.nx, cache.ny, cache.nz)), sigma=config.filter_sigma
        ).ravel()
        rho_phys, drho_proj = heaviside_projection(rho_bar, beta=beta, eta=config.eta_proj)
        rho_phys = np.clip(rho_phys, rho_floor, 1.0)
        if has_passive:
            rho_phys = enforce_passive_regions(rho_phys, cache, solid_value=1.0)

        # --- constitutive evaluation ---
        C_local, dC_local = local_builder(rho_phys, C0)
        C_global, dC_global = rotate_local_family_to_global(C_local, dC_local, Q_m)

        eigs = np.linalg.eigvalsh(C_global)
        local_min = eigs[:, 0]
        worst_local_eig = min(worst_local_eig, float(np.min(local_min)))
        max_inad_frac = max(max_inad_frac, float(np.mean(local_min < -1.0e-12)))

        # --- solve ---
        coeffs = C_global[:, cache.pair_i, cache.pair_j]
        dcoeffs = dC_global[:, cache.pair_i, cache.pair_j]

        K = assemble_global_stiffness_from_basis(cache, coeffs)
        Kff = K[cache.free, :][:, cache.free]

        U = np.zeros(cache.ndof, dtype=float)
        U[cache.free] = spsolve(Kff, cache.F[cache.free])

        # --- sensitivity ---
        Ue = U[cache.edof]
        q = np.einsum('ei,kij,ej->ek', Ue, cache.H_basis, Ue, optimize=True)

        dJ_drhophys = -np.sum(dcoeffs * q, axis=1)
        dJ_drhobar = dJ_drhophys * drho_proj
        dc = apply_filter(
            dJ_drhobar.reshape((cache.nx, cache.ny, cache.nz)),
            sigma=config.filter_sigma,
        ).ravel()

        # Track positive-sensitivity fraction
        if has_passive:
            pos_sens_frac = float(np.mean(dc[cache.design_mask] > 0.0))
        else:
            pos_sens_frac = float(np.mean(dc > 0.0))
        max_pos_sens_frac = max(max_pos_sens_frac, pos_sens_frac)

        # --- sensitivity treatment ---
        if has_passive:
            dc_eff = dc.copy()
            dc_eff[cache.design_mask] = np.minimum(dc_eff[cache.design_mask], -1.0e-12)
            dc_eff[~cache.design_mask] = 0.0
        else:
            if method == 'spectral':
                dc = np.minimum(dc, -1.0e-12)
            dc_eff = dc

        # --- volume sensitivity ---
        dV_drhobar = drho_proj / cache.nel
        dv = apply_filter(
            dV_drhobar.reshape((cache.nx, cache.ny, cache.nz)),
            sigma=config.filter_sigma,
        ).ravel()
        dv = np.maximum(dv, 1.0e-12)
        if has_passive:
            dv[~cache.design_mask] = 0.0

        # --- OC update ---
        l1, l2 = 0.0, 1.0e6
        rho_candidate = rho.copy()

        if has_passive:
            dm = cache.design_mask
            while l2 - l1 > 1.0e-5:
                lmid = 0.5 * (l1 + l2)
                ratio = np.maximum(-dc_eff[dm] / (lmid * dv[dm]), 1.0e-16)
                rho_candidate = rho.copy()
                rho_candidate[dm] = rho[dm] * np.sqrt(ratio)
                rho_candidate[dm] = np.clip(rho_candidate[dm], rho[dm] - config.move, rho[dm] + config.move)
                rho_candidate[dm] = np.clip(rho_candidate[dm], rho_floor, 1.0)
                rho_candidate = enforce_passive_regions(rho_candidate, cache, solid_value=1.0)

                rho_bar_c = apply_filter(
                    rho_candidate.reshape((cache.nx, cache.ny, cache.nz)),
                    sigma=config.filter_sigma,
                ).ravel()
                rho_phys_c, _ = heaviside_projection(rho_bar_c, beta=beta, eta=config.eta_proj)
                rho_phys_c = enforce_passive_regions(np.clip(rho_phys_c, rho_floor, 1.0), cache)

                if np.mean(rho_phys_c) > config.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid
        else:
            while l2 - l1 > 1.0e-5:
                lmid = 0.5 * (l1 + l2)
                ratio = np.maximum(-dc_eff / (lmid * dv), 1.0e-16)
                rho_candidate = rho * np.sqrt(ratio)
                rho_candidate = np.clip(rho_candidate, rho - config.move, rho + config.move)
                rho_candidate = np.clip(rho_candidate, rho_floor, 1.0)

                rho_bar_c = apply_filter(
                    rho_candidate.reshape((cache.nx, cache.ny, cache.nz)),
                    sigma=config.filter_sigma,
                ).ravel()
                rho_phys_c, _ = heaviside_projection(rho_bar_c, beta=beta, eta=config.eta_proj)

                if np.mean(rho_phys_c) > config.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid

        rho = rho_candidate

    if verbose:
        print(" " * 80, end="\r")

    # --- Final state ---
    rho_bar_final = apply_filter(
        rho.reshape((cache.nx, cache.ny, cache.nz)),
        sigma=config.filter_sigma,
    ).ravel()
    rho_phys_final, _ = heaviside_projection(
        rho_bar_final, beta=config.beta_final, eta=config.eta_proj,
    )
    rho_phys_final = np.clip(rho_phys_final, rho_floor, 1.0)
    if has_passive:
        rho_phys_final = enforce_passive_regions(rho_phys_final, cache, solid_value=1.0)

    # Re-evaluate compliance
    C_local_f, _ = local_builder(rho_phys_final, C0)
    C_global_f, _ = rotate_local_family_to_global(
        C_local_f, np.zeros_like(C_local_f), Q_m,
    )
    coeffs_f = C_global_f[:, cache.pair_i, cache.pair_j]
    K_f = assemble_global_stiffness_from_basis(cache, coeffs_f)
    Kff_f = K_f[cache.free, :][:, cache.free]
    U_f = np.zeros(cache.ndof, dtype=float)
    U_f[cache.free] = spsolve(Kff_f, cache.F[cache.free])
    final_compliance = float(cache.F @ U_f)

    eigs_f = np.linalg.eigvalsh(C_global_f)
    final_min_eig = float(np.min(eigs_f[:, 0]))
    final_inad_frac = float(np.mean(eigs_f[:, 0] < -1.0e-12))

    return OptResult(
        rho_phys=rho_phys_final.reshape((cache.nx, cache.ny, cache.nz)),
        rho_raw=rho.copy(),
        final_compliance=final_compliance,
        final_volume=float(np.mean(rho_phys_final)),
        final_min_eig=final_min_eig,
        final_inad_frac=final_inad_frac,
        worst_local_eig=worst_local_eig,
        max_inad_frac=max_inad_frac,
        max_pos_sens_frac=max_pos_sens_frac,
    )


# ---------------------------------------------------------------------------
# Post-hoc evaluation under a different constitutive law
# ---------------------------------------------------------------------------
def evaluate_design(
    theta_deg: float,
    rho_phys_grid: np.ndarray,
    cache,
    eval_method: str = "spectral",
    params: MaterialParams = DEFAULT_PARAMS,
) -> dict:
    """
    Re-evaluate a final topology under a (possibly different) constitutive law.

    Returns dict with keys: eval_method, compliance, min_eig, inad_frac, volume.
    """
    C0 = isotropic_voigt(params.E0, params.nu0)
    R = rotation_about_y(theta_deg)
    Q_m = mandel_rotation_matrix_from_R(R)
    builder = constitutive_builder(eval_method, params)

    rho_phys = rho_phys_grid.ravel()
    rho_phys = np.clip(rho_phys, params.rho_floor, 1.0)
    if _has_passive(cache):
        rho_phys = enforce_passive_regions(rho_phys, cache, solid_value=1.0)

    C_local, _ = builder(rho_phys, C0)
    dummy_dC = np.zeros_like(C_local)
    C_global, _ = rotate_local_family_to_global(C_local, dummy_dC, Q_m)

    coeffs = C_global[:, cache.pair_i, cache.pair_j]
    K = assemble_global_stiffness_from_basis(cache, coeffs)
    Kff = K[cache.free, :][:, cache.free]

    U = np.zeros(cache.ndof, dtype=float)
    U[cache.free] = spsolve(Kff, cache.F[cache.free])

    eigs = np.linalg.eigvalsh(C_global)
    return {
        "eval_method": eval_method,
        "compliance": float(cache.F @ U),
        "min_eig": float(np.min(eigs[:, 0])),
        "inad_frac": float(np.mean(eigs[:, 0] < -1.0e-12)),
        "volume": float(np.mean(rho_phys)),
    }
