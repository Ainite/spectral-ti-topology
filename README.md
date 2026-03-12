# Congruence-Based TI Topology Optimization

Companion code for:

> **F. Yang**, "Algebraic Coercivity in Transversely Isotropic Topology Optimization for Additive Manufacturing: A Kelvin--Mandel Spectral Framework," 
> 2026.

This repository reproduces all numerical experiments (Figures 3–6,
Tables 1–3) reported in the paper. The codebase is organized as a
lightweight Python library (`spectral_ti`) with thin per-figure
generation scripts.

# Spectral TI Topology Optimization

Reproducible code for the paper:

> **Algebraic Coercivity in Transversely Isotropic Topology Optimization for Additive Manufacturing: A Kelvin--Mandel Spectral Framework**

## Quick Start

```bash
# Install
pip install -e .

# Generate all figures and tables (quick mode, ~minutes)
cd scripts
python generate_all.py

# Full manuscript quality (~hours)
python generate_all.py --manuscript
```

## Repository Structure

```
src/spectral_ti/          Core library
  ├── constitutive.py     Material models (isotropic, spectral, ICP)
  ├── mandel.py           Mandel notation & rotation utilities
  ├── fem.py              Finite element engine (Kronecker + Hex8)
  ├── optimization.py     OC topology optimization driver
  ├── diagnostics.py      Eigenvalue trajectories & admissibility
  └── visualization.py    3D iso-surface rendering & academic style

scripts/                   Figure & table generators
  ├── generate_figure3.py  Constitutive autopsy (eigenvalue + modulus surfaces)
  ├── generate_figure4.py  Algorithmic superiority (MMS + forward benchmark)
  ├── generate_figure5.py  3D cantilever comparison
  ├── generate_figure6.py  3D MBB beam comparison
  ├── generate_table1.py   MMS convergence data
  ├── generate_table2.py   Cantilever quantitative metrics
  ├── generate_table3.py   MBB quantitative metrics (reads Figure 6 output)
  └── generate_all.py      Run everything
```

## Module Design

The codebase is organized around a shared physics engine with thin script
wrappers:

| Module | Purpose | Used by |
|--------|---------|---------|
| `constitutive` | All material interpolation laws | Every script |
| `mandel` | 6×6 rotation algebra | Fig 5, 6; Tab 2, 3 |
| `fem` | Two FE methods (Kronecker + standard Hex8) | Every script |
| `optimization` | Unified OC loop with `OptConfig` | Fig 5, 6; Tab 2 |
| `diagnostics` | Eigenvalue analysis, density benchmarks | Fig 3, 4 |
| `visualization` | 3D plotting, academic style | Fig 3, 4, 5, 6 |

### Two Constitutive Paths

1. **Spectral / congruence** — `local_spectral_ti()`, `congruence_ti_from_base()`:
   algebraically guaranteed positive-definite for all ρ ∈ (0, 1].

2. **Baseline ICP** — `local_baseline_ti()`, `baseline_componentwise_tensor()`:
   component-wise penalization that may lose positive-definiteness at low density.

### Two FE Implementations

1. **Kronecker product** (Table 1, Figure 4a): efficient for constant-coefficient
   problems on the unit cube.

2. **Standard Hex8 with operator basis** (Figure 5, 6; Table 2): supports
   variable coefficients and rotated anisotropy.

## Individual Scripts

Each script can be run independently:

```bash
cd scripts

# Quick runs
python generate_figure3.py
python generate_table1.py --profile quick
python generate_figure4.py --profile quick
python generate_figure5.py
python generate_table2.py --profile quick
python generate_figure6.py
python generate_table3.py --input-csv Figure6_MBB_metrics.csv
```

## Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.9
- Matplotlib ≥ 3.6
- scikit-image ≥ 0.19

## License

See LICENSE file.
