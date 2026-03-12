"""
Finite element engine for 3D structured hexahedral meshes.

Two assembly strategies coexist:

1. **Kronecker product method** — for constant-coefficient problems on a unit
   cube with homogeneous Dirichlet BCs (Table 1 MMS / Figure 4 panel a).
2. **Standard Hex8 assembly with operator basis** — for variable-coefficient
   problems with rotated anisotropy (Figure 5 / Figure 6 / Table 2).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix


# =============================================================================
# Part A: Kronecker product FEM (constant coefficient, unit cube)
# =============================================================================

def one_dimensional_matrices(N: int) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """1-D mass, stiffness, and gradient matrices for N elements on [0, 1]."""
    h = 1.0 / N
    Me = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    Ke = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    Ge = 0.5 * np.array([[-1.0, -1.0], [1.0, 1.0]])

    n = N + 1
    rows, cols = [], []
    data_M, data_K, data_G = [], [], []
    for e in range(N):
        conn = [e, e + 1]
        for a in range(2):
            for b in range(2):
                rows.append(conn[a])
                cols.append(conn[b])
                data_M.append(Me[a, b])
                data_K.append(Ke[a, b])
                data_G.append(Ge[a, b])
    M = sp.coo_matrix((data_M, (rows, cols)), shape=(n, n)).tocsr()
    K = sp.coo_matrix((data_K, (rows, cols)), shape=(n, n)).tocsr()
    G = sp.coo_matrix((data_G, (rows, cols)), shape=(n, n)).tocsr()
    return M, K, G


def derivative_operators(N: int) -> Dict[Tuple[int, int], sp.csr_matrix]:
    """9 Kronecker-product derivative operators A[(alpha, beta)]."""
    M1, K1, G1 = one_dimensional_matrices(N)
    GT1 = G1.T.tocsr()

    def factor(dim, alpha, beta):
        if dim == alpha and dim == beta:
            return K1
        if dim == alpha and dim != beta:
            return G1
        if dim == beta and dim != alpha:
            return GT1
        return M1

    A = {}
    for alpha in range(3):
        for beta in range(3):
            Fx = factor(0, alpha, beta)
            Fy = factor(1, alpha, beta)
            Fz = factor(2, alpha, beta)
            A[(alpha, beta)] = sp.kron(Fz, sp.kron(Fy, Fx, format="csr"), format="csr")
    return A


def gradient_to_engineering_strain_map() -> np.ndarray:
    """9→6 mapping from displacement gradient to engineering strain."""
    S = np.zeros((6, 9), dtype=float)
    S[0, 0] = 1.0
    S[1, 4] = 1.0
    S[2, 8] = 1.0
    S[3, 5] = 1.0; S[3, 7] = 1.0
    S[4, 2] = 1.0; S[4, 6] = 1.0
    S[5, 1] = 1.0; S[5, 3] = 1.0
    return S


def build_constant_elasticity_matrix(N: int, C: np.ndarray) -> sp.csr_matrix:
    """Assemble the global stiffness matrix via Kronecker products (constant C)."""
    A = derivative_operators(N)
    S = gradient_to_engineering_strain_map()
    D = S.T @ C @ S  # 9×9

    n_scalar = (N + 1) ** 3
    blocks = [[None] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            K_ab = sp.csr_matrix((n_scalar, n_scalar), dtype=float)
            for alpha in range(3):
                for beta in range(3):
                    coeff = D[3 * a + alpha, 3 * b + beta]
                    if abs(coeff) > 0.0:
                        K_ab = K_ab + coeff * A[(alpha, beta)]
            blocks[a][b] = K_ab.tocsr()
    return sp.bmat(blocks, format="csr")


# -- MMS helpers --

def assemble_1d_load_vector(
    N: int, func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """1-D load vector via Gauss quadrature."""
    h = 1.0 / N
    n = N + 1
    vec = np.zeros(n, dtype=float)
    gp, gw = np.polynomial.legendre.leggauss(5)
    for e in range(N):
        x0 = e * h
        x1 = (e + 1) * h
        xm = 0.5 * (x0 + x1)
        xr = 0.5 * (x1 - x0)
        xq = xm + xr * gp
        wq = xr * gw
        N1 = (x1 - xq) / h
        N2 = (xq - x0) / h
        fq = func(xq)
        vec[e] += np.sum(wq * N1 * fq)
        vec[e + 1] += np.sum(wq * N2 * fq)
    return vec


def kron3(vz, vy, vx):
    return np.kron(vz, np.kron(vy, vx))


def mms_force_term_coefficients(C: np.ndarray) -> Dict[str, np.ndarray]:
    """MMS body-force coefficients for the sinusoidal exact solution."""
    a = 1.0 / np.sqrt(3.0)
    pi2 = np.pi**2

    def d_eps_dx(hxx, hyy, hzz, hyz, hxz, hxy):
        return np.array([a * hxx, a * hxy, a * hxz,
                         a * (hxz + hxy), a * (hxz + hxx), a * (hxy + hxx)])

    def d_eps_dy(hxx, hyy, hzz, hyz, hxz, hxy):
        return np.array([a * hxy, a * hyy, a * hyz,
                         a * (hyz + hyy), a * (hxz + hxy), a * (hyy + hxy)])

    def d_eps_dz(hxx, hyy, hzz, hyz, hxz, hxy):
        return np.array([a * hxz, a * hyz, a * hzz,
                         a * (hzz + hyz), a * (hzz + hxz), a * (hyz + hxz)])

    def body_force_from_h(h):
        hxx, hyy, hzz, hyz, hxz, hxy = h
        sx = C @ d_eps_dx(hxx, hyy, hzz, hyz, hxz, hxy)
        sy = C @ d_eps_dy(hxx, hyy, hzz, hyz, hxz, hxy)
        sz = C @ d_eps_dz(hxx, hyy, hzz, hyz, hxz, hxy)
        return np.array([
            -(sx[0] + sy[5] + sz[4]),
            -(sx[5] + sy[1] + sz[3]),
            -(sx[4] + sy[3] + sz[2]),
        ])

    return {
        "sss": body_force_from_h(np.array([-pi2, -pi2, -pi2, 0.0, 0.0, 0.0])),
        "ccs": body_force_from_h(np.array([0.0, 0.0, 0.0, 0.0, 0.0, +pi2])),
        "csc": body_force_from_h(np.array([0.0, 0.0, 0.0, 0.0, +pi2, 0.0])),
        "scc": body_force_from_h(np.array([0.0, 0.0, 0.0, +pi2, 0.0, 0.0])),
    }


def build_mms_rhs(N: int, C: np.ndarray) -> np.ndarray:
    """Assemble the MMS right-hand side."""
    coeffs = mms_force_term_coefficients(C)
    s = lambda x: np.sin(np.pi * x)
    c = lambda x: np.cos(np.pi * x)

    ls = assemble_1d_load_vector(N, s)
    lc = assemble_1d_load_vector(N, c)

    term_vectors = {
        "sss": kron3(ls, ls, ls),
        "ccs": kron3(ls, lc, lc),
        "csc": kron3(lc, ls, lc),
        "scc": kron3(lc, lc, ls),
    }

    n_scalar = (N + 1) ** 3
    rhs = np.zeros(3 * n_scalar, dtype=float)
    for term_name, coeff in coeffs.items():
        term = term_vectors[term_name]
        for comp in range(3):
            rhs[comp * n_scalar: (comp + 1) * n_scalar] += coeff[comp] * term
    return rhs


def boundary_node_mask(N: int) -> np.ndarray:
    n1 = N + 1
    ix, iy, iz = np.meshgrid(np.arange(n1), np.arange(n1), np.arange(n1), indexing="ij")
    mask = (ix == 0) | (ix == N) | (iy == 0) | (iy == N) | (iz == 0) | (iz == N)
    return mask.ravel(order="F")


def exact_mms_solution_vector(N: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, N + 1)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    phi = phi.ravel(order="F")
    e = 1.0 / np.sqrt(3.0)
    return np.concatenate([e * phi, e * phi, e * phi])


def solve_zero_dirichlet(
    K: sp.csr_matrix,
    rhs: np.ndarray,
    boundary_scalar_mask: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve Ku = f with homogeneous Dirichlet BCs via CG."""
    n_scalar = boundary_scalar_mask.size
    free_scalar = np.where(~boundary_scalar_mask)[0]
    free = np.concatenate([free_scalar, free_scalar + n_scalar, free_scalar + 2 * n_scalar])
    Kff = K[free][:, free].tocsr()
    bf = rhs[free]

    Mdiag = Kff.diagonal().copy()
    Mdiag[np.abs(Mdiag) < 1e-14] = 1.0
    M = spla.LinearOperator(Kff.shape, lambda x: x / Mdiag)
    uf, info = spla.cg(Kff, bf, rtol=tol, atol=0.0, maxiter=5000, M=M)
    if info != 0:
        raise RuntimeError(f"CG failed with info={info}")

    u = np.zeros(K.shape[0], dtype=float)
    u[free] = uf
    return u


# =============================================================================
# Part B: Standard Hex8 assembly with operator basis (variable coefficient)
# =============================================================================

def get_B_matrices(dx: float, dy: float, dz: float) -> List[np.ndarray]:
    """Strain-displacement matrices at 2×2×2 Gauss points for a Hex8 element."""
    pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    Bs = []
    inv_dx, inv_dy, inv_dz = 2.0 / dx, 2.0 / dy, 2.0 / dz

    for t in pts:
        for s in pts:
            for r in pts:
                dNdr = np.array([
                    -0.125 * (1 - s) * (1 - t),  0.125 * (1 - s) * (1 - t),
                     0.125 * (1 + s) * (1 - t), -0.125 * (1 + s) * (1 - t),
                    -0.125 * (1 - s) * (1 + t),  0.125 * (1 - s) * (1 + t),
                     0.125 * (1 + s) * (1 + t), -0.125 * (1 + s) * (1 + t),
                ])
                dNds = np.array([
                    -0.125 * (1 - r) * (1 - t), -0.125 * (1 + r) * (1 - t),
                     0.125 * (1 + r) * (1 - t),  0.125 * (1 - r) * (1 - t),
                    -0.125 * (1 - r) * (1 + t), -0.125 * (1 + r) * (1 + t),
                     0.125 * (1 + r) * (1 + t),  0.125 * (1 - r) * (1 + t),
                ])
                dNdt = np.array([
                    -0.125 * (1 - r) * (1 - s), -0.125 * (1 + r) * (1 - s),
                    -0.125 * (1 + r) * (1 + s), -0.125 * (1 - r) * (1 + s),
                     0.125 * (1 - r) * (1 - s),  0.125 * (1 + r) * (1 - s),
                     0.125 * (1 + r) * (1 + s),  0.125 * (1 - r) * (1 + s),
                ])

                B = np.zeros((6, 24), dtype=float)
                for i in range(8):
                    col = 3 * i
                    B[0, col + 0] = dNdr[i] * inv_dx
                    B[1, col + 1] = dNds[i] * inv_dy
                    B[2, col + 2] = dNdt[i] * inv_dz
                    B[3, col + 1] = dNdt[i] * inv_dz
                    B[3, col + 2] = dNds[i] * inv_dy
                    B[4, col + 0] = dNdt[i] * inv_dz
                    B[4, col + 2] = dNdr[i] * inv_dx
                    B[5, col + 0] = dNds[i] * inv_dy
                    B[5, col + 1] = dNdr[i] * inv_dx
                Bs.append(B)
    return Bs


def build_mesh_connectivity(nx: int, ny: int, nz: int) -> np.ndarray:
    """Element-DOF connectivity for a structured hex mesh."""
    nel = nx * ny * nz
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    n1 = ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz
    n2 = (ix + 1) * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz
    n3 = (ix + 1) * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz
    n4 = ix * (ny + 1) * (nz + 1) + (iy + 1) * (nz + 1) + iz

    edof = np.zeros((nel, 24), dtype=int)
    nodes = np.stack([n1, n2, n3, n4, n1 + 1, n2 + 1, n3 + 1, n4 + 1], axis=-1).reshape(-1, 8)
    for i in range(8):
        edof[:, 3 * i:3 * i + 3] = nodes[:, i:i + 1] * 3 + np.array([0, 1, 2])
    return edof


def build_operator_basis(Bs: List[np.ndarray], vol_g: float):
    """Pre-compute the 21 symmetric element-stiffness basis matrices."""
    pairs = []
    for i in range(6):
        for j in range(i, 6):
            pairs.append((i, j))

    H = np.zeros((len(pairs), 24, 24), dtype=float)
    for k, (i, j) in enumerate(pairs):
        Eij = np.zeros((6, 6), dtype=float)
        if i == j:
            Eij[i, j] = 1.0
        else:
            Eij[i, j] = 1.0
            Eij[j, i] = 1.0
        Hk = np.zeros((24, 24), dtype=float)
        for B in Bs:
            Hk += B.T @ Eij @ B * vol_g
        H[k] = Hk

    pair_i = np.array([p[0] for p in pairs], dtype=int)
    pair_j = np.array([p[1] for p in pairs], dtype=int)
    return H, pair_i, pair_j


def assemble_global_stiffness_from_basis(cache, coeffs: np.ndarray):
    """Assemble global K from per-element upper-triangle coefficients."""
    Ke_all = np.einsum('ek,kij->eij', coeffs, cache.H_basis, optimize=True)
    data = Ke_all.reshape(-1)
    return coo_matrix(
        (data, (cache.rows, cache.cols)),
        shape=(cache.ndof, cache.ndof),
    ).tocsc()


# ---------------------------------------------------------------------------
# FE cache data containers
# ---------------------------------------------------------------------------
@dataclass
class FECache:
    """Standard cantilever FE cache."""
    nx: int
    ny: int
    nz: int
    nel: int
    ndof: int
    edof: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    H_basis: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    free: np.ndarray
    F: np.ndarray


@dataclass
class FECacheMBB(FECache):
    """MBB-specific FE cache with passive-region support."""
    support_anchor_mask: np.ndarray = None
    load_anchor_mask: np.ndarray = None
    passive_solid_mask: np.ndarray = None
    design_mask: np.ndarray = None
    support_nodes: list = None
    load_node: int = 0


# ---------------------------------------------------------------------------
# Cache builders
# ---------------------------------------------------------------------------
def _build_common_cache(nx, ny, nz):
    """Shared setup for both cantilever and MBB caches."""
    nel = nx * ny * nz
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    ndof = 3 * n_nodes

    edof = build_mesh_connectivity(nx, ny, nz)
    Bs = get_B_matrices(1.0, 1.0, 1.0)
    vol_g = 1.0 / 8.0
    H_basis, pair_i, pair_j = build_operator_basis(Bs, vol_g)

    rows = np.repeat(edof, 24, axis=1).ravel()
    cols = np.tile(edof, 24).ravel()

    return nel, n_nodes, ndof, edof, H_basis, pair_i, pair_j, rows, cols


def build_fe_cache_cantilever(nx: int = 30, ny: int = 10, nz: int = 10) -> FECache:
    """Build FE cache for a cantilever beam (fixed at x=0, loaded at x=nx)."""
    nel, n_nodes, ndof, edof, H_basis, pair_i, pair_j, rows, cols = \
        _build_common_cache(nx, ny, nz)

    node_idx = np.arange(n_nodes)
    node_x = node_idx // ((ny + 1) * (nz + 1))
    node_y = (node_idx % ((ny + 1) * (nz + 1))) // (nz + 1)
    node_z = node_idx % (nz + 1)

    fixed_nodes = np.where(node_x == 0)[0]
    fixed = np.repeat(fixed_nodes * 3, 3) + np.tile([0, 1, 2], fixed_nodes.size)
    free = np.setdiff1d(np.arange(ndof), fixed)

    F = np.zeros(ndof, dtype=float)
    load_nodes = np.where(
        (node_x == nx)
        & (node_y >= ny // 2 - 1) & (node_y <= ny // 2 + 1)
        & (node_z >= nz // 2 - 1) & (node_z <= nz // 2 + 1)
    )[0]
    F[load_nodes * 3 + 2] = -1.0 / load_nodes.size

    return FECache(
        nx=nx, ny=ny, nz=nz, nel=nel, ndof=ndof,
        edof=edof, rows=rows, cols=cols,
        H_basis=H_basis, pair_i=pair_i, pair_j=pair_j,
        free=free, F=F,
    )


def _node_id_mbb(ix, iy, iz, ny, nz):
    return ix * (ny + 1) * (nz + 1) + iy * (nz + 1) + iz


def _fill_box(mask, x0, x1, y0, y1, z0, z1):
    mask[max(0, x0):min(mask.shape[0], x1),
         max(0, y0):min(mask.shape[1], y1),
         max(0, z0):min(mask.shape[2], z1)] = True


def build_fe_cache_mbb(
    nx: int = 32, ny: int = 12, nz: int = 10,
    passive_pad: int = 2, passive_height: int = 2,
) -> FECacheMBB:
    """Build FE cache for the MBB beam with passive regions."""
    nel, n_nodes, ndof, edof, H_basis, pair_i, pair_j, rows, cols = \
        _build_common_cache(nx, ny, nz)

    c0 = _node_id_mbb(0, 0, 0, ny, nz)
    c1 = _node_id_mbb(nx, 0, 0, ny, nz)
    c2 = _node_id_mbb(0, ny, 0, ny, nz)
    c3 = _node_id_mbb(nx, ny, 0, ny, nz)
    support_nodes = [c0, c1, c2, c3]
    load_node = _node_id_mbb(nx // 2, ny // 2, nz, ny, nz)

    fixed_dofs = []
    for n in support_nodes:
        fixed_dofs.append(n * 3 + 2)
    fixed_dofs += [c0 * 3 + 0, c0 * 3 + 1, c1 * 3 + 1, c2 * 3 + 0]
    fixed = np.unique(np.array(fixed_dofs, dtype=int))
    free = np.setdiff1d(np.arange(ndof), fixed)

    F = np.zeros(ndof, dtype=float)
    F[load_node * 3 + 2] = -1.0

    support_anchor_mask = np.zeros((nx, ny, nz), dtype=bool)
    _fill_box(support_anchor_mask, 0, passive_pad, 0, passive_pad, 0, passive_height)
    _fill_box(support_anchor_mask, nx - passive_pad, nx, 0, passive_pad, 0, passive_height)
    _fill_box(support_anchor_mask, 0, passive_pad, ny - passive_pad, ny, 0, passive_height)
    _fill_box(support_anchor_mask, nx - passive_pad, nx, ny - passive_pad, ny, 0, passive_height)

    load_anchor_mask = np.zeros((nx, ny, nz), dtype=bool)
    xc = nx // 2
    yc = ny // 2
    _fill_box(load_anchor_mask, xc - 1, xc + 1, yc - 1, yc + 1, nz - passive_height, nz)

    passive_solid_mask = support_anchor_mask | load_anchor_mask
    design_mask = ~passive_solid_mask.ravel()

    return FECacheMBB(
        nx=nx, ny=ny, nz=nz, nel=nel, ndof=ndof,
        edof=edof, rows=rows, cols=cols,
        H_basis=H_basis, pair_i=pair_i, pair_j=pair_j,
        free=free, F=F,
        support_anchor_mask=support_anchor_mask,
        load_anchor_mask=load_anchor_mask,
        passive_solid_mask=passive_solid_mask,
        design_mask=design_mask,
        support_nodes=support_nodes,
        load_node=load_node,
    )


# =============================================================================
# Part C: Variable-coefficient Hex8 assembly (Figure 4 panels b, c)
# =============================================================================

_SIGNS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=float)


def _node_id_cube(i, j, k, N):
    return i + (N + 1) * (j + (N + 1) * k)


def _hex8_connectivity(ex, ey, ez, N):
    return [
        _node_id_cube(ex, ey, ez, N),
        _node_id_cube(ex + 1, ey, ez, N),
        _node_id_cube(ex + 1, ey + 1, ez, N),
        _node_id_cube(ex, ey + 1, ez, N),
        _node_id_cube(ex, ey, ez + 1, N),
        _node_id_cube(ex + 1, ey, ez + 1, N),
        _node_id_cube(ex + 1, ey + 1, ez + 1, N),
        _node_id_cube(ex, ey + 1, ez + 1, N),
    ]


def _gauss_points_2x2x2():
    g = 1.0 / np.sqrt(3.0)
    pts = []
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                pts.append((xi, eta, zeta, 1.0))
    return pts


def _b_matrix_at(xi, eta, zeta, h):
    dN_dxi = np.zeros((8, 3), dtype=float)
    for a in range(8):
        sx, sy, sz = _SIGNS[a]
        dN_dxi[a, 0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
        dN_dxi[a, 1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
        dN_dxi[a, 2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    dN_dx = dN_dxi * (2.0 / h)

    B = np.zeros((6, 24), dtype=float)
    for a in range(8):
        dNx, dNy, dNz = dN_dx[a]
        c = 3 * a
        B[0, c + 0] = dNx
        B[1, c + 1] = dNy
        B[2, c + 2] = dNz
        B[3, c + 1] = dNz; B[3, c + 2] = dNy
        B[4, c + 0] = dNz; B[4, c + 2] = dNx
        B[5, c + 0] = dNy; B[5, c + 1] = dNx
    return B


def _consistent_body_force_hex8(body_force, h):
    vol = h**3
    fe = np.zeros(24, dtype=float)
    nodal = body_force * vol / 8.0
    for a in range(8):
        fe[3 * a: 3 * a + 3] = nodal
    return fe


def assemble_variable_coefficient_system(
    N: int,
    constitutive: Callable,
    C0: np.ndarray,
    body_force: np.ndarray,
    density_field_func: Callable,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Element-by-element assembly for variable-coefficient problems on [0,1]^3."""
    h = 1.0 / N
    ndof = 3 * (N + 1) ** 3

    row_list, col_list, data_list = [], [], []
    rhs = np.zeros(ndof, dtype=float)

    gps = _gauss_points_2x2x2()
    Bgps = [_b_matrix_at(xi, eta, zeta, h) for xi, eta, zeta, _ in gps]
    detJ = (h / 2.0) ** 3
    fe_const = _consistent_body_force_hex8(body_force, h)

    for ez in range(N):
        zc = (ez + 0.5) * h
        for ey in range(N):
            yc = (ey + 0.5) * h
            for ex in range(N):
                xc = (ex + 0.5) * h
                rho = density_field_func(xc, yc, zc)
                C = constitutive(rho, C0)

                Ke = np.zeros((24, 24), dtype=float)
                for B in Bgps:
                    Ke += B.T @ C @ B * detJ

                conn = _hex8_connectivity(ex, ey, ez, N)
                edofs = np.array([3 * n + d for n in conn for d in range(3)], dtype=int)
                rhs[edofs] += fe_const

                rr, cc = np.meshgrid(edofs, edofs, indexing="ij")
                row_list.extend(rr.ravel().tolist())
                col_list.extend(cc.ravel().tolist())
                data_list.extend(Ke.ravel().tolist())

    K = sp.coo_matrix((data_list, (row_list, col_list)), shape=(ndof, ndof)).tocsr()
    return K, rhs


def fixed_dofs_x0_face(N: int) -> np.ndarray:
    """All DOFs on the x=0 face of an (N+1)^3 grid."""
    n1 = N + 1
    dofs = []
    for k in range(n1):
        for j in range(n1):
            n = _node_id_cube(0, j, k, N)
            dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])
    return np.array(sorted(set(dofs)), dtype=int)


def solve_with_dirichlet_zero(K, rhs, fixed):
    """Sparse direct solve with homogeneous Dirichlet BCs."""
    ndof = rhs.size
    is_free = np.ones(ndof, dtype=bool)
    is_free[fixed] = False
    free = np.where(is_free)[0]
    Kff = K[free][:, free].tocsr()
    uf = spla.spsolve(Kff, rhs[free])
    u = np.zeros(ndof, dtype=float)
    u[free] = uf
    return u


def nodal_coordinates(N: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, N + 1)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
