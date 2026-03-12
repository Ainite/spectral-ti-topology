"""
Generate Table 3: Quantitative comparison for the 3D MBB beam study.

Reads the metrics CSV produced by generate_figure6.py and emits
CSV, JSON, and LaTeX table files.
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Table3Case:
    theta_deg: int
    method: str
    eval_method: str
    J_opt_env: float
    J_eval_spd: float
    volume: float
    final_min_eig: float
    final_inad_frac: float
    worst_local_eig: float
    max_inad_frac: float
    max_pos_sens_frac: float


# ---------------------------------------------------------------------------
# Readers / validators
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "theta_deg", "design_method", "eval_method",
    "J_opt_env", "J_eval_spd", "volume",
    "final_min_eig", "final_inad_frac",
    "worst_local_eig", "max_inad_frac", "max_pos_sens_frac",
]


def read_metrics_csv(path: Path) -> List[Table3Case]:
    if not path.exists():
        raise FileNotFoundError(f"Input metrics file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        cases = []
        for row in reader:
            cases.append(Table3Case(
                theta_deg=int(float(row["theta_deg"])),
                method=row["design_method"].strip().lower(),
                eval_method=row["eval_method"].strip().lower(),
                J_opt_env=float(row["J_opt_env"]),
                J_eval_spd=float(row["J_eval_spd"]),
                volume=float(row["volume"]),
                final_min_eig=float(row["final_min_eig"]),
                final_inad_frac=float(row["final_inad_frac"]),
                worst_local_eig=float(row["worst_local_eig"]),
                max_inad_frac=float(row["max_inad_frac"]),
                max_pos_sens_frac=float(row["max_pos_sens_frac"]),
            ))

    if not cases:
        raise ValueError("CSV contains no data rows.")
    _validate(cases)
    return cases


def _validate(cases):
    expected = {(0, "icp"), (0, "spectral"), (45, "icp"), (45, "spectral")}
    found = {(c.theta_deg, c.method) for c in cases}
    if found != expected:
        raise ValueError(
            f"Expected keys: {sorted(expected)}\nFound: {sorted(found)}"
        )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(cases, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "theta_deg", "method", "eval_method",
            "J_opt_env", "J_eval_spd", "volume",
            "final_min_eig", "final_inad_frac",
            "worst_local_eig", "max_inad_frac", "max_pos_sens_frac",
        ])
        for c in sorted(cases, key=lambda x: (x.theta_deg, x.method)):
            writer.writerow([
                c.theta_deg, c.method, c.eval_method,
                f"{c.J_opt_env:.16e}", f"{c.J_eval_spd:.16e}", f"{c.volume:.16e}",
                f"{c.final_min_eig:.16e}", f"{c.final_inad_frac:.16e}",
                f"{c.worst_local_eig:.16e}", f"{c.max_inad_frac:.16e}",
                f"{c.max_pos_sens_frac:.16e}",
            ])


def write_json(cases, path):
    payload = [asdict(c) for c in sorted(cases, key=lambda x: (x.theta_deg, x.method))]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_latex_table(cases, path):
    rows: Dict[Tuple[int, str], Table3Case] = {(c.theta_deg, c.method): c for c in cases}
    method_map = {"icp": "ICP", "spectral": "Congruence"}

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{"
        r"\textbf{Quantitative comparison for the 3D MBB beam study (Case Study II).} "
        r"The table reports the final common-physics compliance $J_{\mathrm{eval}}$, "
        r"the post-processed physical volume fraction, and constitutive diagnostics "
        r"for the final designs obtained by the baseline ICP method and the present "
        r"congruence framework under two build-direction orientations, "
        r"$\theta = 0^\circ$ and $\theta = 45^\circ$."
        r"}"
    )
    lines.append(r"\label{tab:mbb_quantitative_comparison}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"$\theta$ & Method & Common-Physics Compliance $J_{\mathrm{eval}}$ & "
        r"Post-processed Physical Volume & $\lambda_{\min}^{\mathrm{final}}$ & "
        r"$\phi_{\mathrm{inad}}^{\mathrm{final}}$ \\"
    )
    lines.append(r"\midrule")

    for ang in [0, 45]:
        for i, method in enumerate(["icp", "spectral"]):
            c = rows[(ang, method)]
            theta_str = rf"${ang}^\circ$" if i == 0 else ""
            lines.append(
                f"{theta_str} & {method_map[method]} & "
                f"${c.J_eval_spd:.2f}$ & ${c.volume:.3f}$ & "
                f"${c.final_min_eig:.3e}$ & ${c.final_inad_frac:.3f}$ \\\\"
            )
        if ang != 45:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(cases):
    print("=" * 120)
    print(f"{'Angle':<8} | {'Method':<12} | {'J_eval (SPD)':<14} | "
          f"{'Vol_phys':<9} | {'λ_min(final)':<14} | {'inad(final)':<12}")
    print("-" * 120)
    for c in sorted(cases, key=lambda x: (x.theta_deg, x.method)):
        display = "ICP" if c.method == "icp" else "Congruence"
        print(f"{c.theta_deg:>3}°     | {display:<12} | "
              f"{c.J_eval_spd:>12.4e} | {c.volume:>7.4f} | "
              f"{c.final_min_eig:>12.4e} | {c.final_inad_frac:>10.3%}")
    print("=" * 120)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Table 3 from Figure 6 metrics.")
    parser.add_argument("--input-csv", type=Path, default=Path("Figure6_MBB_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = read_metrics_csv(args.input_csv)
    print_summary(cases)

    write_csv(cases, args.output_dir / "Table3_Compliance_Data.csv")
    write_json(cases, args.output_dir / "Table3_Compliance_Data.json")
    write_latex_table(cases, args.output_dir / "Table3_Compliance_Table.tex")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
