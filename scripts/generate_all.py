"""
Generate all figures and tables for the manuscript.

Usage:
    python scripts/generate_all.py              # quick profile
    python scripts/generate_all.py --manuscript  # full-resolution
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate all paper outputs.")
    parser.add_argument("--manuscript", action="store_true",
                        help="Use manuscript-quality settings (slower)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    profile = "manuscript" if args.manuscript else "quick"

    t0 = time.time()

    print("\n" + "=" * 60)
    print("  Figure 3: Integrated constitutive autopsy")
    print("=" * 60)
    from generate_figure3 import main as gen3
    gen3(output_name=str(out / "Figure3_Integrated_Autopsy.png"))

    print("\n" + "=" * 60)
    print("  Table 1: MMS convergence study")
    print("=" * 60)
    sys.argv = ["", f"--profile={profile}", f"--output-dir={out}"]
    from generate_table1 import main as gen_t1
    gen_t1()

    print("\n" + "=" * 60)
    print("  Figure 4: Algorithmic superiority")
    print("=" * 60)
    sys.argv = ["", f"--profile={profile}", f"--output-dir={out}"]
    from generate_figure4 import main as gen4
    gen4()

    print("\n" + "=" * 60)
    print("  Figure 5: 3D cantilever comparison")
    print("=" * 60)
    from generate_figure5 import main as gen5
    gen5(output_name=str(out / "Figure5_Cantilever.png"))

    print("\n" + "=" * 60)
    print("  Table 2: Cantilever quantitative data")
    print("=" * 60)
    sys.argv = ["", f"--profile={profile}", f"--output-dir={out}"]
    from generate_table2 import main as gen_t2
    gen_t2()

    print("\n" + "=" * 60)
    print("  Figure 6: 3D MBB beam comparison")
    print("=" * 60)
    from generate_figure6 import main as gen6
    gen6(output_name=str(out / "Figure6_MBB_Beam.png"))

    print("\n" + "=" * 60)
    print("  Table 3: MBB quantitative data (from Figure 6 metrics)")
    print("=" * 60)
    sys.argv = ["", f"--input-csv=Figure6_MBB_metrics.csv", f"--output-dir={out}"]
    from generate_table3 import main as gen_t3
    gen_t3()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  All outputs generated in {elapsed:.1f} s")
    print(f"  Output directory: {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
