"""
Mandel notation and rotation utilities for 6×6 elasticity tensors.

Provides the mapping between engineering Voigt and Mandel representations
and the rotation of constitutive families to arbitrary build-direction
orientations via the Bond matrix.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mandel scaling matrices
# ---------------------------------------------------------------------------
_s2 = np.sqrt(2.0)
M_MANDEL = np.diag([1.0, 1.0, 1.0, _s2, _s2, _s2])
M_MANDEL_INV = np.diag([1.0, 1.0, 1.0, 1.0 / _s2, 1.0 / _s2, 1.0 / _s2])


# ---------------------------------------------------------------------------
# Mandel basis tensors (cached at module level)
# ---------------------------------------------------------------------------
def _mandel_basis_tensors() -> np.ndarray:
    B = np.zeros((6, 3, 3), dtype=float)
    B[0, 0, 0] = 1.0
    B[1, 1, 1] = 1.0
    B[2, 2, 2] = 1.0
    B[3, 1, 2] = B[3, 2, 1] = 1.0 / _s2
    B[4, 0, 2] = B[4, 2, 0] = 1.0 / _s2
    B[5, 0, 1] = B[5, 1, 0] = 1.0 / _s2
    return B


MANDEL_BASIS = _mandel_basis_tensors()


# ---------------------------------------------------------------------------
# 6×6 rotation (Bond) matrix from a 3×3 rotation
# ---------------------------------------------------------------------------
def mandel_rotation_matrix_from_R(R: np.ndarray) -> np.ndarray:
    """Compute the 6×6 Mandel rotation matrix Q from a 3×3 rotation R."""
    Q = np.zeros((6, 6), dtype=float)
    for j in range(6):
        T_rot = R @ MANDEL_BASIS[j] @ R.T
        for i in range(6):
            Q[i, j] = np.tensordot(MANDEL_BASIS[i], T_rot)
    return Q


# ---------------------------------------------------------------------------
# Build-direction rotation about the y-axis
# ---------------------------------------------------------------------------
def rotation_about_y(theta_deg: float) -> np.ndarray:
    """3×3 rotation about the y-axis by *theta_deg* degrees."""
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c],
    ])


# ---------------------------------------------------------------------------
# Rotate a vectorized family of local tensors to the global frame
# ---------------------------------------------------------------------------
def rotate_local_family_to_global(
    C_local_eng: np.ndarray,
    dC_local_eng: np.ndarray,
    Q_m: np.ndarray,
) -> tuple:
    """
    Rotate (nel, 6, 6) engineering-Voigt local tensors to the global frame.

    Parameters
    ----------
    C_local_eng : (nel, 6, 6)  local constitutive tensors
    dC_local_eng : (nel, 6, 6)  local constitutive derivatives
    Q_m : (6, 6)  Mandel rotation matrix

    Returns
    -------
    C_global_eng, dC_global_eng : each (nel, 6, 6)
    """
    # Local engineering -> local Mandel
    C_local_m = np.einsum(
        'ab,ebc,cd->ead', M_MANDEL, C_local_eng, M_MANDEL, optimize=True)
    dC_local_m = np.einsum(
        'ab,ebc,cd->ead', M_MANDEL, dC_local_eng, M_MANDEL, optimize=True)

    # Rotate in Mandel space
    C_global_m = np.einsum(
        'ab,ebc,dc->ead', Q_m, C_local_m, Q_m, optimize=True)
    dC_global_m = np.einsum(
        'ab,ebc,dc->ead', Q_m, dC_local_m, Q_m, optimize=True)

    # Global Mandel -> global engineering
    C_global_eng = np.einsum(
        'ab,ebc,cd->ead', M_MANDEL_INV, C_global_m, M_MANDEL_INV, optimize=True)
    dC_global_eng = np.einsum(
        'ab,ebc,cd->ead', M_MANDEL_INV, dC_global_m, M_MANDEL_INV, optimize=True)
    return C_global_eng, dC_global_eng
