"""
Generate Figure 5: 3D cantilever topology comparison.

Multi-angle (0°, 45°, 90°) side-by-side comparison of
baseline ICP vs. spectral framework topologies.
"""

import time
import warnings

import matplotlib.pyplot as plt
from scipy.sparse.linalg import MatrixRankWarning

from spectral_ti.fem import build_fe_cache_cantilever
from spectral_ti.optimization import run_optimization, OptConfig
from spectral_ti.visualization import (
    academic_style, plot_3d_topology, add_build_direction_arrow_3d,
)

warnings.filterwarnings("ignore", category=MatrixRankWarning)


def main(output_name="Figure5_Cantilever.png"):
    academic_style()

    cache = build_fe_cache_cantilever(nx=30, ny=10, nz=10)
    config = OptConfig(
        volfrac=0.17, max_iter=36, filter_sigma=1.10, move=0.08,
        beta_final=5.5,
        beta_milestones=(0.25, 0.50, 0.75),
        beta_values=(1.0, 2.0, 3.5),
    )

    angles = [0, 45, 90]
    methods = ["icp", "spectral"]

    print("=" * 120)
    print(f"{'Condition':<14} | {'Method':<10} | {'worst local λ_min':<24} | "
          f"{'max inadmissible frac':<22} | {'max pos-sensitivity frac':<24}")
    print("-" * 120)

    results = {}
    t0 = time.time()

    for theta in angles:
        for method in methods:
            out = run_optimization(
                theta_deg=theta, method=method, cache=cache, config=config,
            )
            results[(theta, method)] = out.rho_phys

            tag = "non-SPD" if out.worst_local_eig < 0.0 else "SPD"
            print(f"{theta:>3}° aligned   | {method.upper():<10} | "
                  f"{out.worst_local_eig:+.4e} ({tag:<7}) | "
                  f"{out.max_inad_frac:>8.3%}              | "
                  f"{out.max_pos_sens_frac:>8.3%}")

    print("=" * 120)
    print(f"Optimization finished in {time.time() - t0:.1f} s")

    # --- Plot ---
    fig = plt.figure(figsize=(14, 12))
    c_icp = [0.75, 0.75, 0.75]
    c_spec = [0.27, 0.51, 0.71]

    for i, theta in enumerate(angles):
        ax_icp = fig.add_subplot(3, 2, 2 * i + 1, projection="3d")
        plot_3d_topology(
            ax_icp, results[(theta, "icp")], c_icp,
            fr"Baseline ICP ($\theta = {theta}^\circ$)",
            iso_level=0.27, view_mode="cantilever",
        )
        add_build_direction_arrow_3d(ax_icp, theta)

        ax_spec = fig.add_subplot(3, 2, 2 * i + 2, projection="3d")
        plot_3d_topology(
            ax_spec, results[(theta, "spectral")], c_spec,
            fr"Spectral Framework ($\theta = {theta}^\circ$)",
            iso_level=0.27, view_mode="cantilever",
        )
        add_build_direction_arrow_3d(ax_spec, theta)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_name, format="png", dpi=300, bbox_inches="tight")
    print(f"Generated: {output_name}")


if __name__ == "__main__":
    main()
