#!/usr/bin/env python3
"""plot_results.py  Task 2B plots: speedup and instruction count vs. SIMD width & size.

Reads the two CSVs produced in this directory:
  - simd_width_results.csv   (from `./bin/simd_width_bench`)        size,variant,time_ms,gflops,speedup
  - instruction_counts.csv   (from `./run_perf_sweep.sh`)           size,variant,reps,instructions,instructions_per_rep,correct

and writes PNGs into ../plots/ (the submission's top-level plots/ folder):
  - task2_simd_width_speedup_vs_size.png       one line per SIMD width, x = matrix size
  - task2_simd_width_speedup_vs_width.png      one line per matrix size, x = SIMD width
  - task2_simd_width_instructions_vs_size.png  one line per SIMD width, x = matrix size (log-y)

Only pure stdlib csv + matplotlib -- no pandas required.

Usage:
    python3 plot_results.py
    python3 plot_results.py --results other.csv --instructions other2.csv --outdir ../plots
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT_ORDER = ["sse128", "avx2_256", "avx512"]
VARIANT_LABEL = {"sse128": "128-bit (SSE)", "avx2_256": "256-bit (AVX2)", "avx512": "512-bit (AVX-512)"}
VARIANT_WIDTH = {"sse128": 128, "avx2_256": 256, "avx512": 512}


def read_csv_rows(path):
    if not os.path.exists(path):
        print(f"warning: {path} not found, skipping plots that need it", file=sys.stderr)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def plot_speedup_vs_size(rows, outdir):
    # variant -> sorted list of (size, speedup)
    by_variant = {}
    for row in rows:
        v = row["variant"]
        if v not in VARIANT_ORDER:
            continue
        sp = to_float(row.get("speedup"))
        size = to_float(row.get("size"))
        if sp is None or size is None:
            continue
        by_variant.setdefault(v, []).append((size, sp))

    if not by_variant:
        print("no usable rows for speedup_vs_size", file=sys.stderr)
        return

    plt.figure(figsize=(7, 5))
    for v in VARIANT_ORDER:
        if v not in by_variant:
            continue
        pts = sorted(by_variant[v])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=VARIANT_LABEL[v])

    plt.xscale("log", base=2)
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("Speedup over matmul_naive")
    plt.title("SIMD speedup vs. matrix size, by SIMD width")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(outdir, "task2_simd_width_speedup_vs_size.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_speedup_vs_width(rows, outdir):
    # size -> sorted list of (width, speedup)
    by_size = {}
    for row in rows:
        v = row["variant"]
        if v not in VARIANT_ORDER:
            continue
        sp = to_float(row.get("speedup"))
        size = to_float(row.get("size"))
        if sp is None or size is None:
            continue
        by_size.setdefault(size, []).append((VARIANT_WIDTH[v], sp))

    if not by_size:
        print("no usable rows for speedup_vs_width", file=sys.stderr)
        return

    plt.figure(figsize=(7, 5))
    for size in sorted(by_size):
        pts = sorted(by_size[size])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"size={int(size)}")

    plt.xlabel("SIMD register width (bits)")
    plt.ylabel("Speedup over matmul_naive")
    plt.title("SIMD speedup vs. SIMD width, by matrix size")
    plt.xticks(sorted({VARIANT_WIDTH[v] for v in VARIANT_ORDER}))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(outdir, "task2_simd_width_speedup_vs_width.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_instructions_vs_size(rows, outdir):
    by_variant = {}
    for row in rows:
        v = row["variant"]
        if v not in VARIANT_ORDER:
            continue
        instr = to_float(row.get("instructions_per_rep"))
        size = to_float(row.get("size"))
        if instr is None or size is None:
            continue
        by_variant.setdefault(v, []).append((size, instr))

    if not by_variant:
        print("no usable rows for instructions_vs_size (run run_perf_sweep.sh first)",
              file=sys.stderr)
        return

    plt.figure(figsize=(7, 5))
    for v in VARIANT_ORDER:
        if v not in by_variant:
            continue
        pts = sorted(by_variant[v])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=VARIANT_LABEL[v])

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("Instructions per call (log scale)")
    plt.title("Instruction count vs. matrix size, by SIMD width")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(outdir, "task2_simd_width_instructions_vs_size.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "simd_width_results.csv"))
    ap.add_argument("--instructions", default=os.path.join(here, "instruction_counts.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "..", "plots"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results_rows = read_csv_rows(args.results)
    if results_rows:
        plot_speedup_vs_size(results_rows, args.outdir)
        plot_speedup_vs_width(results_rows, args.outdir)

    instr_rows = read_csv_rows(args.instructions)
    if instr_rows:
        plot_instructions_vs_size(instr_rows, args.outdir)


if __name__ == "__main__":
    main()
