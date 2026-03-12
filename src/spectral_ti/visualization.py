"""
Visualization utilities for topology optimization results.

Provides 3D iso-surface rendering (marching cubes), directional Young-modulus
surfaces, build-direction arrows, and a unified academic style.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter


def academic_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


# =============================================================================
# 3D topology iso-surface rendering
# =============================================================================

def remove_structurally_irrelevant_components(mask, min_voxels=8):
    """Keep only mechanically meaningful connected components (cantilever)."""
    labeled, ncomp = ndi.label(mask)
    if ncomp == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = False

    nx = mask.shape[0]
    for lab in range(1, ncomp + 1):
        comp = (labeled == lab)
        touches_support = np.any(comp[0, :, :])
        touches_load = np.any(comp[nx - 1, :, :])
        large_enough = counts[lab] >= min_voxels
        if (touches_support or touches_load) and large_enough:
            keep[lab] = True

    if not np.any(keep[1:]):
        if len(counts) > 1:
            keep[np.argmax(counts[1:]) + 1] = True

    return keep[labeled]


def select_load_to_support_component(
    binary_mask, support_anchor_mask, load_anchor_mask, dilation_iter=2,
):
    """Keep only the component bridging load and support (MBB)."""
    if not np.any(binary_mask):
        return binary_mask

    support_anchor = ndi.binary_dilation(support_anchor_mask, iterations=dilation_iter)
    load_anchor = ndi.binary_dilation(load_anchor_mask, iterations=dilation_iter)

    labeled, ncomp = ndi.label(binary_mask)
    if ncomp == 0:
        return binary_mask

    counts = np.bincount(labeled.ravel())
    load_labels = np.unique(labeled[load_anchor & (labeled > 0)])

    keep = np.zeros(ncomp + 1, dtype=bool)
    for lab in load_labels:
        comp = labeled == lab
        if np.any(comp & support_anchor):
            keep[lab] = True

    if np.any(keep[1:]):
        return keep[labeled]

    if load_labels.size > 0:
        best_lab = load_labels[np.argmax(counts[load_labels])]
        return labeled == best_lab

    support_labels = np.unique(labeled[support_anchor & (labeled > 0)])
    if support_labels.size > 0:
        best_lab = support_labels[np.argmax(counts[support_labels])]
        return labeled == best_lab

    best_lab = np.argmax(counts[1:]) + 1
    return labeled == best_lab


def clean_volume_for_plot(vol_data, iso_level=0.27, cache=None, min_voxels=8):
    """
    Smooth and clean a voxel field for iso-surface extraction.

    If *cache* is an MBB cache (has support/load anchor masks), use the
    load-to-support component filter; otherwise use the cantilever filter.
    """
    vol_smooth = gaussian_filter(vol_data, sigma=0.50, mode='reflect')
    mask = vol_smooth >= iso_level

    if cache is not None and hasattr(cache, 'support_anchor_mask') and cache.support_anchor_mask is not None:
        mask = ndi.binary_closing(mask, structure=np.ones((2, 2, 2), dtype=bool))
        mask = ndi.binary_opening(mask, structure=np.ones((2, 2, 2), dtype=bool))
        keep_mask = select_load_to_support_component(
            mask,
            support_anchor_mask=cache.support_anchor_mask,
            load_anchor_mask=cache.load_anchor_mask,
            dilation_iter=2,
        )
        return np.where(keep_mask, vol_smooth, 0.0)
    else:
        mask = remove_structurally_irrelevant_components(mask, min_voxels=min_voxels)
        return np.where(mask, vol_smooth, 0.0)


def plot_3d_topology(ax, vol_data, color, title, iso_level=0.27,
                     cache=None, view_mode="cantilever"):
    """
    Render a 3D iso-surface on a Matplotlib Axes3D.

    Parameters
    ----------
    view_mode : str
        "cantilever" or "mbb" — controls padding and view angle.
    """
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vol_plot = clean_volume_for_plot(vol_data, iso_level=iso_level, cache=cache)

    try:
        verts, faces, _, _ = measure.marching_cubes(vol_plot, level=iso_level)
    except Exception:
        vol_plot = np.maximum(vol_data, 0.0)
        verts, faces, _, _ = measure.marching_cubes(vol_plot, level=0.25)

    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('none')

    ls = LightSource(azdeg=315, altdeg=45)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1.0e-12)
    intensity = ls.shade_normals(normals, fraction=0.8)

    rgb = np.array(color, dtype=float)
    final_colors = np.clip(rgb * (0.20 + 0.80 * intensity[:, None]), 0.0, 1.0)
    mesh.set_facecolor(final_colors)
    ax.add_collection3d(mesh)

    nx, ny, nz = vol_data.shape

    if view_mode == "mbb":
        _set_tight_mesh_view(ax, verts, zoom=1.05, elev=18, azim=-62)
    else:
        padding_y = 6.0
        padding_z = 6.0
        ax.set_xlim(0, nx)
        ax.set_ylim(-padding_y, ny)
        ax.set_zlim(-padding_z, nz)
        ax.set_box_aspect((nx, ny + padding_y, nz + padding_z))
        ax.view_init(elev=15, azim=-75)

    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=5, fontsize=12)


def _set_tight_mesh_view(ax, verts, zoom=1.05, elev=18, azim=-62):
    """Tight bounding-box view for MBB plots."""
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(np.maximum(maxs - mins, 1.0))

    half_x = 0.50 * span / zoom
    half_y = 0.34 * span / zoom
    half_z = 0.24 * span / zoom

    ax.set_xlim(center[0] - half_x, center[0] + half_x)
    ax.set_ylim(center[1] - half_y, center[1] + half_y)
    ax.set_zlim(center[2] - half_z, center[2] + half_z)
    ax.set_box_aspect((1.75, 1.00, 0.62))
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type('persp', focal_length=1.25)
    except Exception:
        pass


# =============================================================================
# Build-direction arrow
# =============================================================================

def add_build_direction_arrow_3d(ax, theta_deg, start=(15.0, -4.0, -4.0), length=6.0):
    """Add a 3D quiver arrow for the build direction (cantilever style)."""
    theta = np.radians(theta_deg)
    dx = length * np.sin(theta)
    dz = length * np.cos(theta)
    ax.quiver(
        start[0], start[1], start[2],
        dx, 0.0, dz,
        color='#D62728', linewidth=2.5, arrow_length_ratio=0.3, zorder=100,
    )
    ax.text(
        start[0] + dx, start[1], start[2] + dz + 0.5,
        r" $\mathbf{n}$", color='#D62728',
        fontweight='bold', fontsize=13, zorder=100,
    )


def add_build_direction_overlay(fig, ax, theta_deg):
    """Add a 2D overlay arrow for the build direction (MBB style)."""
    bbox = ax.get_position()
    overlay = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height], frameon=False)
    overlay.set_xlim(0.0, 1.0)
    overlay.set_ylim(0.0, 1.0)
    overlay.axis('off')
    overlay.set_zorder(1000)
    overlay.patch.set_alpha(0.0)

    theta = np.radians(theta_deg)
    start = np.array([0.43, 0.08])
    vec = 0.11 * np.array([np.sin(theta), np.cos(theta)])
    end = start + vec
    overlay.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='simple', fc='#d62728', ec='#d62728', lw=0.0, alpha=0.95),
        annotation_clip=False,
    )
    overlay.text(end[0] + 0.010, end[1] + 0.006, 'n',
                 color='#d62728', fontsize=13, fontweight='bold')


# =============================================================================
# Helpers for directional Young-modulus 3D surfaces
# =============================================================================
def set_equal_limits(ax, lim, elev=18, azim=-45):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)


def get_global_limit(*arrays, margin=1.08, min_window=0.08):
    finite_vals = []
    for arr in arrays:
        vals = arr[np.isfinite(arr)]
        if vals.size > 0:
            finite_vals.append(vals.ravel())
    if not finite_vals:
        return 1.0
    all_vals = np.concatenate(finite_vals)
    lim = np.max(np.abs(all_vals))
    return max(lim * margin, min_window)


# =============================================================================
# Slope reference line  (Figure 4 panel a)
# =============================================================================
def make_slope_reference(x, y, order=2.0, anchor_index=1):
    anchor_index = max(0, min(anchor_index, len(x) - 1))
    x0 = x[anchor_index]
    y0 = y[anchor_index]
    y_ref = y0 * (x0 / x) ** order
    return x, y_ref
