"""
Generate Table 2: Quantitative comparison for the 3D cantilever study.

Runs the same optimization as Figure 5 and reports compliance, volume,
and constitutive diagnostics.
"""

import argparse
import csv
import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning

from spectral_ti.constitutive import isotropic_voigt, DEFAULT_PARAMS
from spectral_ti.mandel import rotation_about_y, mandel_rotation_matrix_from_R, rotate_local_family_to_global
from spectral_ti.fem import build_fe_cache_cantilever, assemble_global_stiffness_from_basis
from spectral_ti.optimization import run_optimization, evaluate_design, OptConfig

warnings.filterwarnings("ignore", category=MatrixRankWarning)


@dataclass
class Table2Case:
    angle_deg: int
    method: str
    compliance: float
    physical_volume_fraction: float
    final_min_lambda: float
    final_inadmissible_fraction: float
    worst_lambda_over_history: float
    max_inadmissible_fraction_over_history: float
    max_pos_sensitivity_fraction: float
    ndof: int
    nel: int
    iterations: int


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(cases: List[Table2Case], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "angle_deg", "method", "compliance", "physical_volume_fraction",
            "final_min_lambda", "final_inadmissible_fraction",
            "worst_lambda_over_history", "max_inadmissible_fraction_over_history",
            "max_pos_sensitivity_fraction", "ndof", "nel", "iterations",
        ])
        for c in cases:
            writer.writerow([
                c.angle_deg, c.method,
                f"{c.compliance:.16e}", f"{c.physical_volume_fraction:.16e}",
                f"{c.final_min_lambda:.16e}", f"{c.final_inadmissible_fraction:.16e}",
                f"{c.worst_lambda_over_history:.16e}",
                f"{c.max_inadmissible_fraction_over_history:.16e}",
                f"{c.max_pos_sensitivity_fraction:.16e}",
                c.ndof, c.nel, c.iterations,
            ])


def write_json(cases: List[Table2Case], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in cases], indent=2), encoding="utf-8")


def write_latex_table(cases: List[Table2Case], path: Path) -> None:
    rows: Dict[Tuple[int, str], Table2Case] = {(c.angle_deg, c.method): c for c in cases}
    method_map = {"ICP": "ICP", "SPECTRAL": "Congruence"}
    lines = []
    lines.append(r"\begin{tabular}{c c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{$\theta$} & \multirow{2}{*}{Method} & Common-Physics "
        r"& \multirow{2}{*}{Post-processed Physical Volume} "
        r"& \multirow{2}{*}{$\lambda_{\min}^{\mathrm{final}}$} "
        r"& \multirow{2}{*}{$\phi_{\mathrm{inad}}^{\mathrm{final}}$} "
        r"& \multirow{2}{*}{$\lambda_{\min}^{\mathrm{worst}}$} "
        r"& \multirow{2}{*}{$\phi_{\mathrm{inad}}^{\max}$} \\"
    )
    lines.append(r" &  & Compliance $J_{\mathrm{eval}}$ &  &  &  &  &  \\")
    lines.append(r"\midrule")

    for ang in [0, 45, 90]:
        for i, method in enumerate(["ICP", "SPECTRAL"]):
            c = rows[(ang, method)]
            theta_str = rf"${ang}^\circ$" if i == 0 else ""
            lines.append(
                f"{theta_str} & {method_map[method]} & "
                f"${c.compliance:.3e}$ & ${c.physical_volume_fraction:.3f}$ & "
                f"${c.final_min_lambda:.3e}$ & ${c.final_inadmissible_fraction:.3f}$ & "
                f"${c.worst_lambda_over_history:.3e}$ & "
                f"${c.max_inadmissible_fraction_over_history:.3f}$ \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Table 2 data.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--profile", choices=["quick", "manuscript"], default="manuscript")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.profile == "quick":
        cache = build_fe_cache_cantilever(nx=24, ny=8, nz=8)
        config = OptConfig(volfrac=0.17, max_iter=24, filter_sigma=1.05, move=0.10,
                           beta_final=5.5, beta_milestones=(0.25, 0.50, 0.75),
                           beta_values=(1.0, 2.0, 3.5))
    else:
        cache = build_fe_cache_cantilever(nx=30, ny=10, nz=10)
        config = OptConfig(volfrac=0.17, max_iter=36, filter_sigma=1.10, move=0.08,
                           beta_final=5.5, beta_milestones=(0.25, 0.50, 0.75),
                           beta_values=(1.0, 2.0, 3.5))

    cases: List[Table2Case] = []
    t0 = time.time()
    print("Generating Table 2 data...")
    print("=" * 140)

    for theta in [0, 45, 90]:
        for method in ["icp", "spectral"]:
            out = run_optimization(
                theta_deg=theta, method=method, cache=cache, config=config,
            )
            # Evaluate under unified spectral physics
            ev = evaluate_design(theta, out.rho_phys, cache, eval_method="spectral")

            cases.append(Table2Case(
                angle_deg=theta,
                method=method.upper(),
                compliance=ev["compliance"],
                physical_volume_fraction=out.final_volume,
                final_min_lambda=out.final_min_eig,
                final_inadmissible_fraction=out.final_inad_frac,
                worst_lambda_over_history=out.worst_local_eig,
                max_inadmissible_fraction_over_history=out.max_inad_frac,
                max_pos_sensitivity_fraction=out.max_pos_sens_frac,
                ndof=cache.ndof,
                nel=cache.nel,
                iterations=config.max_iter,
            ))
            display = "ICP" if method == "icp" else "Congruence"
            print(f"  {theta:>3}° | {display:<12} | J={ev['compliance']:.4e} | "
                  f"vol={out.final_volume:.4f}")

    print("=" * 140)
    print(f"Finished in {time.time() - t0:.1f} s")

    write_csv(cases, args.output_dir / "Table2_Compliance_Data.csv")
    write_json(cases, args.output_dir / "Table2_Compliance_Data.json")
    write_latex_table(cases, args.output_dir / "Table2_Compliance_Table.tex")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
