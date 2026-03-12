"""
Generate Table 1: MMS convergence study under the congruence-based TI formulation.

Outputs CSV, JSON, and LaTeX table files.
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from spectral_ti.constitutive import (
    isotropic_voigt, congruence_ti_from_base,
    calibrate_eta_family_for_xi, anisotropy_ratio,
)
from spectral_ti.fem import (
    build_constant_elasticity_matrix, build_mms_rhs,
    boundary_node_mask, exact_mms_solution_vector, solve_zero_dirichlet,
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class MMSCase:
    xi_target: float
    xi_effective: float
    eta_L: float
    eta_T: float
    eta_S: float
    N: int
    dof: int
    rel_l2_error: float
    rate: Optional[float]


# ---------------------------------------------------------------------------
# MMS driver
# ---------------------------------------------------------------------------
def run_mms_cases(
    N_values: Sequence[int],
    xi_list: Sequence[float],
    rho0: float = 0.5,
    p: float = 3.0,
    E0: float = 1.0,
    nu0: float = 0.3,
) -> List[MMSCase]:
    C0 = isotropic_voigt(E0, nu0)
    uex_cache: Dict[int, np.ndarray] = {}
    K_cache = {}
    rhs_cache = {}

    cases: List[MMSCase] = []
    for xi in xi_list:
        eta_L, eta_T, eta_S, xi_eff = calibrate_eta_family_for_xi(xi, C0=C0, rho0=rho0, p=p)
        C_eff = congruence_ti_from_base(C0, rho=rho0, p=p, eta_L=eta_L, eta_T=eta_T, eta_S=eta_S)
        prev_N, prev_err = None, None
        for N in N_values:
            key = (N, float(xi))
            if N not in uex_cache:
                uex_cache[N] = exact_mms_solution_vector(N)
            if key not in K_cache:
                K_cache[key] = build_constant_elasticity_matrix(N, C_eff)
                rhs_cache[key] = build_mms_rhs(N, C_eff)
            bmask = boundary_node_mask(N)
            uh = solve_zero_dirichlet(K_cache[key], rhs_cache[key], bmask)
            uex = uex_cache[N]
            rel_err = float(np.linalg.norm(uh - uex) / np.linalg.norm(uex))
            rate = None if prev_N is None else float(np.log(prev_err / rel_err) / np.log(N / prev_N))
            cases.append(MMSCase(
                xi_target=float(xi), xi_effective=float(xi_eff),
                eta_L=float(eta_L), eta_T=float(eta_T), eta_S=float(eta_S),
                N=int(N), dof=int(3 * (N + 1) ** 3), rel_l2_error=rel_err, rate=rate,
            ))
            prev_N, prev_err = N, rel_err
    return cases


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(cases: Sequence[MMSCase], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["xi_target", "xi_effective", "eta_L", "eta_T", "eta_S",
                          "N", "dof", "rel_l2_error", "rate"])
        for c in cases:
            writer.writerow([
                c.xi_target, c.xi_effective, c.eta_L, c.eta_T, c.eta_S,
                c.N, c.dof, f"{c.rel_l2_error:.16e}",
                "" if c.rate is None else f"{c.rate:.8f}",
            ])


def write_json(cases: Sequence[MMSCase], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in cases], indent=2), encoding="utf-8")


def write_latex_table(cases, path, xi_list, N_values):
    rows = {(c.N, c.xi_target): c for c in cases}
    lines = []
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Resolution} & \multicolumn{4}{c}{\textbf{Relative discrete $\ell_2$ Error}} & \textbf{Rate $r$} \\")
    lines.append(r"\cmidrule(lr){2-5}")
    lines.append(r"$N$ & $\xi = 1$ & $\xi = 10$ & $\xi = 100$ & $\xi = 1000$ & ($\xi=1$) \\")
    lines.append(r"\midrule")
    for N in N_values:
        vals = [rows[(N, float(xi))].rel_l2_error for xi in xi_list]
        rate = rows[(N, float(xi_list[0]))].rate
        rate_str = "---" if rate is None else f"{rate:.2f}"
        val_strs = [f"${v:.2e}$" for v in vals]
        lines.append(f"{N} & " + " & ".join(val_strs) + f" & {rate_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


def grouped_summary(cases):
    grouped: Dict[float, List[MMSCase]] = {}
    for c in cases:
        grouped.setdefault(c.xi_target, []).append(c)
    lines = []
    for xi in sorted(grouped):
        block = sorted(grouped[xi], key=lambda z: z.N)
        first = block[0]
        lines.append(
            f"xi_target={xi:.0f}, xi_effective={first.xi_effective:.4f}, "
            f"eta_L={first.eta_L:.6f}, eta_T={first.eta_T:.6f}, eta_S={first.eta_S:.6f}"
        )
        for c in block:
            rate_str = "---" if c.rate is None else f"{c.rate:.3f}"
            lines.append(f"  N={c.N:>3d}, dof={c.dof:>7d}, rel_l2_error={c.rel_l2_error:.6e}, rate={rate_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Table 1 MMS data.")
    parser.add_argument("--profile", choices=["quick", "manuscript"], default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    N_values = [8, 16, 24, 32] if args.profile == "quick" else [8, 16, 24, 32, 40]
    xi_list = [1.0, 10.0, 100.0, 1000.0]

    cases = run_mms_cases(N_values=N_values, xi_list=xi_list)
    print(grouped_summary(cases))

    write_csv(cases, args.output_dir / "table1_mms_data.csv")
    write_json(cases, args.output_dir / "table1_mms_data.json")
    write_latex_table(cases, args.output_dir / "table1_mms_table.tex", xi_list, N_values)
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
