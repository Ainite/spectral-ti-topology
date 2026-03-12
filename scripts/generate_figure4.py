"""
Generate Figure 4: Algorithmic superiority.

  (a) MMS convergence under congruence-based TI
  (b) Forward self-convergence with localized admissibility loss
  (c) Accuracy vs degrees of freedom
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from spectral_ti.constitutive import (
    isotropic_voigt, baseline_componentwise_tensor, spectral_congruence_tensor,
    MaterialParams, DEFAULT_PARAMS,
)
from spectral_ti.fem import (
    assemble_variable_coefficient_system, fixed_dofs_x0_face,
    solve_with_dirichlet_zero, nodal_coordinates,
)
from spectral_ti.diagnostics import (
    benchmark_density_parameters, density_field, constitutive_admissibility_scan,
    DensityBenchmarkParams,
)
from spectral_ti.visualization import academic_style, make_slope_reference

# Re-use Table 1 MMS driver
from generate_table1 import run_mms_cases, write_csv as write_mms_csv


# ---------------------------------------------------------------------------
# Forward self-convergence benchmark
# ---------------------------------------------------------------------------
@dataclass
class ForwardBenchmarkCase:
    model: str
    N: int
    dof: int
    rel_disc_error: float
    inadmissible_fraction: float
    min_local_lambda: float


def _interpolate_reference(u_ref, N_ref, N_coarse):
    x_ref = np.linspace(0.0, 1.0, N_ref + 1)
    points = nodal_coordinates(N_coarse)
    ncoarse_scalar = (N_coarse + 1) ** 3
    out = np.zeros(3 * ncoarse_scalar, dtype=float)
    for comp in range(3):
        arr = u_ref[comp::3].reshape((N_ref + 1, N_ref + 1, N_ref + 1), order="F")
        interp = RegularGridInterpolator(
            (x_ref, x_ref, x_ref), arr, bounds_error=False, fill_value=None,
        )
        out[comp::3] = interp(points)
    return out


def run_self_convergence_benchmark(
    N_values: Sequence[int],
    N_ref: int,
    body_force=np.array([0.2, -0.1, -1.0]),
    params: MaterialParams = DEFAULT_PARAMS,
) -> Tuple[List[ForwardBenchmarkCase], DensityBenchmarkParams]:
    C0 = isotropic_voigt(params.E0, params.nu0)
    dp = benchmark_density_parameters(C0, params)
    df_func = lambda xc, yc, zc: density_field(xc, yc, zc, dp)

    models = {
        "Baseline ICP": lambda rho, C0_: baseline_componentwise_tensor(rho, C0_, params),
        "Spectral FEM": lambda rho, C0_: spectral_congruence_tensor(rho, C0_),
    }

    results: List[ForwardBenchmarkCase] = []
    for model_name, law in models.items():
        K_ref, rhs_ref = assemble_variable_coefficient_system(
            N_ref, law, C0, body_force, df_func,
        )
        u_ref = solve_with_dirichlet_zero(K_ref, rhs_ref, fixed_dofs_x0_face(N_ref))

        for N in N_values:
            K, rhs = assemble_variable_coefficient_system(N, law, C0, body_force, df_func)
            uh = solve_with_dirichlet_zero(K, rhs, fixed_dofs_x0_face(N))
            u_ref_interp = _interpolate_reference(u_ref, N_ref=N_ref, N_coarse=N)

            rel_err = float(np.linalg.norm(uh - u_ref_interp) / np.linalg.norm(u_ref_interp))
            inad_frac, min_lam = constitutive_admissibility_scan(N, law, C0, dp)

            results.append(ForwardBenchmarkCase(
                model=model_name, N=int(N), dof=int(3 * (N + 1) ** 3),
                rel_disc_error=float(rel_err),
                inadmissible_fraction=float(inad_frac),
                min_local_lambda=float(min_lam),
            ))
    return results, dp


def write_forward_csv(cases, dp, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rho_crit", dp.rho_crit])
        writer.writerow(["density_center", dp.center])
        writer.writerow(["density_amplitude", dp.amplitude])
        writer.writerow(["density_rho_min", dp.rho_min])
        writer.writerow(["density_rho_max", dp.rho_max])
        writer.writerow([])
        writer.writerow(["model", "N", "dof", "rel_disc_error",
                          "inadmissible_fraction", "min_local_lambda"])
        for c in cases:
            writer.writerow([c.model, c.N, c.dof,
                             f"{c.rel_disc_error:.16e}",
                             f"{c.inadmissible_fraction:.16e}",
                             f"{c.min_local_lambda:.16e}"])


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def generate_figure4(mms_cases, forward_cases, dp, output_path):
    academic_style()
    fig = plt.figure(figsize=(14, 6.9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.18)

    # (a) MMS convergence
    ax_a = fig.add_subplot(gs[0])
    xi_list = sorted({c.xi_target for c in mms_cases})
    styles = {
        1.0:    {"m": "o", "c": "#1f77b4", "l": r"Isotropic ($\xi=1$)"},
        10.0:   {"m": "s", "c": "#ff7f0e", "l": r"Moderate ($\xi=10$)"},
        100.0:  {"m": "^", "c": "#2ca02c", "l": r"Strong ($\xi=100$)"},
        1000.0: {"m": "D", "c": "#d62728", "l": r"Extreme ($\xi=1000$)"},
    }
    curves = {}
    for xi in xi_list:
        block = sorted([c for c in mms_cases if c.xi_target == xi], key=lambda z: z.N)
        Ns = np.array([c.N for c in block], dtype=float)
        errs = np.array([c.rel_l2_error for c in block], dtype=float)
        curves[xi] = (Ns, errs)
        st = styles[xi]
        ax_a.loglog(Ns, errs, marker=st["m"], color=st["c"], label=st["l"])

    Ns_iso, errs_iso = curves[1.0]
    ref_x, ref_y = make_slope_reference(Ns_iso, errs_iso, order=2.0,
                                         anchor_index=min(1, len(Ns_iso) - 1))
    ax_a.loglog(ref_x, ref_y, "k:", linewidth=2.4, label=r"Slope reference $O(h^2)$")
    ax_a.set_xlabel(r"Spatial Resolution ($N$)")
    ax_a.set_ylabel(r"Relative $L_2$ Error Norm")
    ax_a.set_title(r"(a) MMS Convergence under the Congruence-Based TI Formulation",
                   fontweight="bold", pad=12)
    ax_a.legend(frameon=False, fontsize=10)
    ax_a.grid(True, which="both", alpha=0.3)

    # (b, c) forward benchmark
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.34)
    ax_b = fig.add_subplot(gs_right[0])
    ax_c = fig.add_subplot(gs_right[1])

    grouped: Dict[str, List[ForwardBenchmarkCase]] = {}
    for c in forward_cases:
        grouped.setdefault(c.model, []).append(c)

    baseline = sorted(grouped["Baseline ICP"], key=lambda z: z.N)
    spectral = sorted(grouped["Spectral FEM"], key=lambda z: z.N)

    N_b = np.array([c.N for c in baseline], dtype=float)
    E_b = np.array([c.rel_disc_error for c in baseline], dtype=float)
    F_b = np.array([c.inadmissible_fraction for c in baseline], dtype=float)
    N_s = np.array([c.N for c in spectral], dtype=float)
    E_s = np.array([c.rel_disc_error for c in spectral], dtype=float)
    F_s = np.array([c.inadmissible_fraction for c in spectral], dtype=float)

    h1 = ax_b.loglog(N_b, E_b, "s--", color="#777777", label="Baseline ICP", zorder=3)
    h2 = ax_b.loglog(N_s, E_s, "o-", color="#0D47A1", label="Spectral FEM", zorder=4)

    plateau = np.median(E_b[-min(2, len(E_b)):])
    ax_b.axhspan(0.92 * plateau, 1.08 * plateau, color="#FFCDD2", alpha=0.30, zorder=1)
    ax_b.annotate(
        "baseline plateau coincides with\nlocalized inadmissibility",
        xy=(N_b[-2], plateau), xycoords="data",
        xytext=(0.08, 0.78), textcoords="axes fraction",
        color="#8B1E1E", fontsize=9.2, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D9A6A6", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#8B1E1E", lw=1.3,
                        shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=0.0"),
        zorder=6,
    )
    ax_b.set_xlabel(r"Spatial Resolution ($N$)")
    ax_b.set_ylabel(r"Relative discrete $\ell_2$ Error")
    ax_b.set_title(r"(b) Forward self-convergence and localized admissibility loss",
                   fontweight="bold", fontsize=14, pad=8)
    ax_b.grid(True, which="both", alpha=0.28)

    ax_b2 = ax_b.twinx()
    h3 = ax_b2.semilogx(N_b, F_b, "d:", color="crimson", linewidth=2.0, markersize=5.5,
                         label=r"Baseline inadmissible fraction", zorder=3)
    h4 = ax_b2.semilogx(N_s, F_s, "x:", color="forestgreen", linewidth=1.8, markersize=5.2,
                         label=r"Spectral inadmissible fraction", zorder=2)
    ax_b2.set_ylabel(r"Inadmissible element fraction", color="crimson")
    ax_b2.tick_params(axis="y", colors="crimson")
    ax_b2.set_ylim(0.0, max(0.35, 1.15 * np.max(F_b)))

    handles = h1 + h2 + h3 + h4
    ax_b.legend(handles, [h.get_label() for h in handles],
                fontsize=9.6, loc="upper right", frameon=True, fancybox=True,
                framealpha=0.92, facecolor="white", edgecolor="#CCCCCC")

    # (c) accuracy vs DOF
    dof_b = np.array([c.dof for c in baseline], dtype=float)
    dof_s = np.array([c.dof for c in spectral], dtype=float)
    ax_c.loglog(dof_b, 1.0 / E_b, "s--", color="#777777", label="Baseline ICP")
    ax_c.loglog(dof_s, 1.0 / E_s, "o-", color="#0D47A1", label="Spectral FEM")
    ax_c.set_xlabel(r"Degrees of Freedom (cost proxy)")
    ax_c.set_ylabel(r"Accuracy ($1 / E_{\ell_2}$)")
    ax_c.set_title(r"(c) Self-convergence accuracy versus degrees of freedom",
                   fontweight="bold", pad=10)
    ax_c.legend(frameon=False, loc="upper left", fontsize=10)
    ax_c.grid(True, which="both", alpha=0.3)

    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.98)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Wrote figure: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4.")
    parser.add_argument("--profile", choices=["quick", "manuscript"], default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mms_N = [8, 16, 24, 32] if args.profile == "quick" else [8, 16, 24, 32, 40]
    forward_N = [6, 8, 10, 12]
    N_ref = 16
    xi_list = [1.0, 10.0, 100.0, 1000.0]

    mms_cases = run_mms_cases(N_values=mms_N, xi_list=xi_list)
    forward_cases, dp = run_self_convergence_benchmark(N_values=forward_N, N_ref=N_ref)

    write_mms_csv(mms_cases, args.output_dir / "figure4_panel_a_mms.csv")
    write_forward_csv(forward_cases, dp, args.output_dir / "figure4_panels_bc_forward.csv")
    generate_figure4(mms_cases, forward_cases, dp,
                     args.output_dir / "Figure4_Algorithmic_Superiority.png")


if __name__ == "__main__":
    main()
