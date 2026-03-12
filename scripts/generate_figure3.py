"""
Generate Figure 3: Integrated constitutive autopsy.

  (a) Coupled normal-mode eigenvalue trajectory
  (b) Directional modulus surface — baseline ICP
  (c) Directional modulus surface — spectral coupling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LightSource

from spectral_ti.constitutive import build_ti_voigt_from_path
from spectral_ti.diagnostics import coupled_normal_block_min_eigs, compute_directional_young_surface
from spectral_ti.visualization import academic_style, set_equal_limits, get_global_limit


def main(output_name="Figure3_Integrated_Autopsy.png"):
    academic_style()

    fig = plt.figure(figsize=(14.2, 7.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.12], wspace=0.06)

    # ------ (a) Eigenvalue trajectory ------
    ax1 = fig.add_subplot(gs[0])

    rhos = np.linspace(0.01, 1.0, 500)
    p1, p3, pc = 3.0, 5.0, 3.0

    min_eig_icp, min_eig_spec = coupled_normal_block_min_eigs(
        rhos, p1=p1, p3=p3, pc=pc,
    )

    ax1.axhspan(-0.15, 0.0, color='red', alpha=0.10,
                label=r'Loss of positivity in coupled normal block ($\lambda_{\min}<0$)')
    ax1.axhspan(0.0, 0.4, color='green', alpha=0.10,
                label=r'Positive coupled normal block ($\lambda_{\min}>0$)')
    ax1.plot(rhos, min_eig_icp, 'r--', linewidth=2.5,
             label='Baseline ICP coupling\n(empirical mixed-exponent penalization)')
    ax1.plot(rhos, min_eig_spec, color='green', linewidth=2.5,
             label='Spectral coupling\n(geometric-mean exponent in the coupled mode)')
    ax1.axhline(0.0, color='black', linewidth=1.0)
    ax1.set_xlim(0.01, 1.0)
    ax1.set_ylim(-0.10, 0.30)
    ax1.set_xlabel(r'Relative Density ($\tilde{\rho}$)', fontsize=14)
    ax1.set_ylabel(r'Minimum Eigenvalue of Coupled Normal Block', fontsize=13)
    ax1.set_title('(a) Coupled Normal-Mode Eigenvalue Trajectory',
                  fontsize=15, fontweight='bold', pad=14)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=10.5, framealpha=0.92)

    # ------ (b, c) Directional modulus surfaces ------
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.13)

    theta = np.linspace(0.0, 2.0 * np.pi, 220)
    phi = np.linspace(0.0, np.pi, 120)
    THETA, PHI = np.meshgrid(theta, phi)

    ls = LightSource(azdeg=315, altdeg=45)
    rho_probe = 0.35
    p_coup_spec = 0.5 * (p1 + p3)
    cap = 2.0

    C_icp = build_ti_voigt_from_path(rho=rho_probe, p1=p1, p3=p3, p13=pc, p44=pc)
    C_spec = build_ti_voigt_from_path(rho=rho_probe, p1=p1, p3=p3, p13=p_coup_spec, p44=p_coup_spec)

    X_icp, Y_icp, Z_icp, R_icp = compute_directional_young_surface(C_icp, THETA, PHI, cap=cap)
    X_spec, Y_spec, Z_spec, R_spec = compute_directional_young_surface(C_spec, THETA, PHI, cap=cap)

    common_lim = get_global_limit(
        X_icp, Y_icp, Z_icp, X_spec, Y_spec, Z_spec, margin=1.05, min_window=0.06,
    )

    common_cmap = cm.viridis
    c_panel_scale = 0.10

    # (b) Baseline ICP
    ax2 = fig.add_subplot(gs_right[0], projection='3d')
    rgb_icp = ls.shade(np.nan_to_num(R_icp, nan=0.0), cmap=common_cmap,
                       vert_exag=0.5, blend_mode='soft')
    ax2.plot_surface(X_icp, Y_icp, Z_icp, facecolors=rgb_icp,
                     rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax2.set_title(
        fr"(b) Directional modulus surface under baseline ICP at $\tilde{{\rho}}={rho_probe:.2f}$",
        fontsize=13, fontweight='bold', y=0.95)
    ax2.axis('off')
    set_equal_limits(ax2, common_lim, elev=18, azim=-45)

    # (c) Spectral
    ax3 = fig.add_subplot(gs_right[1], projection='3d')
    rgb_spec = ls.shade(np.nan_to_num(R_spec, nan=0.0), cmap=common_cmap,
                        vert_exag=0.5, blend_mode='soft')
    ax3.plot_surface(X_spec, Y_spec, Z_spec, facecolors=rgb_spec,
                     rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax3.set_title(
        fr"(c) Directional modulus surface under spectral coupling at $\tilde{{\rho}}={rho_probe:.2f}$",
        fontsize=13, fontweight='bold', y=0.95)
    ax3.axis('off')
    set_equal_limits(ax3, common_lim * c_panel_scale, elev=18, azim=-45)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.985)
    plt.savefig(output_name, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {output_name}")


if __name__ == "__main__":
    main()
