"""
Generate Figure 6: 3D MBB beam topology comparison.

Two-angle (0°, 45°) comparison with passive regions,
plus metrics CSV for Table 3 consumption.
"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import MatrixRankWarning

from spectral_ti.fem import build_fe_cache_mbb
from spectral_ti.optimization import run_optimization, evaluate_design, OptConfig
from spectral_ti.visualization import (
    academic_style, plot_3d_topology, add_build_direction_overlay,
)

warnings.filterwarnings("ignore", category=MatrixRankWarning)


def save_metrics_csv(filename, rows):
    header = (
        'theta_deg,design_method,eval_method,'
        'J_opt_env,J_eval_spd,volume,'
        'final_min_eig,final_inad_frac,worst_local_eig,max_inad_frac,max_pos_sens_frac\n'
    )
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header)
        for r in rows:
            f.write(
                f"{r['theta_deg']},{r['design_method']},{r['eval_method']},"
                f"{r['J_opt_env']:.10e},{r['J_eval_spd']:.10e},{r['volume']:.6f},"
                f"{r['final_min_eig']:.10e},{r['final_inad_frac']:.6f},"
                f"{r['worst_local_eig']:.10e},{r['max_inad_frac']:.6f},"
                f"{r['max_pos_sens_frac']:.6f}\n"
            )


def main(output_name="Figure6_MBB_Beam.png"):
    academic_style()

    cache = build_fe_cache_mbb(nx=32, ny=12, nz=10, passive_pad=2, passive_height=2)
    config = OptConfig(
        volfrac=0.15, max_iter=45, filter_sigma=1.10, move=0.06,
        beta_final=7.0,
        beta_milestones=(0.20, 0.45, 0.70),
        beta_values=(1.0, 2.0, 4.0),
    )

    angles = [0, 45]
    methods = ['icp', 'spectral']

    print('=' * 140)
    print(f"{'Angle':<8} | {'Design':<10} | {'Eval':<10} | "
          f"{'J(opt env)':<14} | {'J(SPD eval)':<14} | {'Vol':<8} | "
          f"{'final λmin':<14} | {'final inad':<12} | "
          f"{'worst λmin':<14} | {'max inad':<10}")
    print('-' * 140)

    results = {}
    metrics_rows = []
    t0 = time.time()

    for theta in angles:
        for method in methods:
            out = run_optimization(
                theta_deg=theta, method=method, cache=cache, config=config,
            )
            eval_spd = evaluate_design(theta, out.rho_phys, cache, eval_method='spectral')
            results[(theta, method)] = out.rho_phys

            metrics = {
                'theta_deg': theta,
                'design_method': method,
                'eval_method': eval_spd['eval_method'],
                'J_opt_env': out.final_compliance,
                'J_eval_spd': eval_spd['compliance'],
                'volume': out.final_volume,
                'final_min_eig': out.final_min_eig,
                'final_inad_frac': out.final_inad_frac,
                'worst_local_eig': out.worst_local_eig,
                'max_inad_frac': out.max_inad_frac,
                'max_pos_sens_frac': out.max_pos_sens_frac,
            }
            metrics_rows.append(metrics)

            print(
                f"{theta:>3}°     | {method.upper():<10} | "
                f"{eval_spd['eval_method'].upper():<10} | "
                f"{metrics['J_opt_env']:<14.4e} | {metrics['J_eval_spd']:<14.4e} | "
                f"{metrics['volume']:<8.4f} | "
                f"{metrics['final_min_eig']:<14.4e} | "
                f"{metrics['final_inad_frac']:<12.3%} | "
                f"{metrics['worst_local_eig']:<14.4e} | "
                f"{metrics['max_inad_frac']:<10.3%}"
            )

    print('=' * 140)
    print(f"Optimization finished in {time.time() - t0:.1f} s")

    save_metrics_csv('Figure6_MBB_metrics.csv', metrics_rows)
    np.savez_compressed(
        'Figure6_MBB_topologies.npz',
        **{f'theta_{theta}_{method}': results[(theta, method)]
           for theta in angles for method in methods},
    )

    # --- Plot ---
    fig = plt.figure(figsize=(13.8, 8.8))
    c_icp = [0.75, 0.75, 0.75]
    c_spec = [0.27, 0.51, 0.71]
    axes = []

    display_iso = {
        ('icp', 0): 0.315, ('icp', 45): 0.315,
        ('spectral', 0): 0.290, ('spectral', 45): 0.285,
    }

    for i, theta in enumerate(angles):
        ax_icp = fig.add_subplot(2, 2, 2 * i + 1, projection='3d')
        plot_3d_topology(
            ax_icp, results[(theta, 'icp')], c_icp,
            fr'Baseline ICP ($\theta={theta}^\circ$)',
            iso_level=display_iso[('icp', theta)],
            cache=cache, view_mode="mbb",
        )
        axes.append((ax_icp, theta))

        ax_spec = fig.add_subplot(2, 2, 2 * i + 2, projection='3d')
        plot_3d_topology(
            ax_spec, results[(theta, 'spectral')], c_spec,
            fr'Spectral Framework ($\theta={theta}^\circ$)',
            iso_level=display_iso[('spectral', theta)],
            cache=cache, view_mode="mbb",
        )
        axes.append((ax_spec, theta))

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.018, top=0.972,
                        wspace=0.08, hspace=0.18)
    fig.canvas.draw()

    for ax, theta in axes:
        add_build_direction_overlay(fig, ax, theta_deg=theta)

    plt.savefig(output_name, format='png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Generated: {output_name}")
    print("Saved: Figure6_MBB_metrics.csv")
    print("Saved: Figure6_MBB_topologies.npz")


if __name__ == '__main__':
    main()
