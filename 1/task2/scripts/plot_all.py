#!/usr/bin/env python3
"""plot_all.py  turns the experiment CSVs in results/ into the report figures in plots/.

Reads whatever experiments have been run and skips the rest, so it is safe to call after
each script rather than only at the end.

  results/b_speedup_vs_size.csv       -> task2_speedup_vs_size.png
                                         task2_gflops_vs_size.png
  results/c_cache_metrics.csv         -> task2_mpki_vs_size.png
  results/d_prefetch_distance.csv     -> task2_prefetch_degree.png
                                         task2_prefetch_degree_l1d.png
  results/e_prefetch_level.csv        -> task2_prefetch_level.png
  results/e_prefetch_level_grid.csv   -> task2_prefetch_level_grid.png
  results/f_hw_prefetcher.csv         -> task2_hw_prefetcher.png
  results/g_simd_width_*.csv          -> task2_simd_width_speedup.png
                                         task2_simd_width_instructions.png
  results/h_block_size.csv            -> task2_block_size.png
  results/i_llama.csv                 -> task2_llama_tokens_per_sec.png

The three filenames the assignment's submission tree names explicitly
(task2_speedup_vs_size.png, task2_prefetch_degree.png, task2_prefetch_level.png) are
produced under exactly those names.

Usage:
    python3 plot_all.py
    python3 plot_all.py --results ../results --outdir ../../plots
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour per stage, held constant across every figure so a reader can track a stage
# from plot to plot without re-reading the legend each time.
STAGE_COLOR = {
    "naive": "#8c8c8c",
    "simd": "#4c78a8",
    "prefetch": "#f58518",
    "optimized": "#54a24b",
}
STAGE_ORDER = ["naive", "simd", "prefetch", "optimized"]

VARIANT_LABEL = {
    "naive": "scalar (naive)",
    "sse128": "128-bit (SSE)",
    "avx2_256": "256-bit (AVX2)",
    "avx512": "512-bit (AVX-512)",
}
VARIANT_ORDER = ["naive", "sse128", "avx2_256", "avx512"]

HINT_ORDER = ["T0", "T1", "T2", "NTA"]


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def finish(fig, ax_or_axes, outdir, name, title, xlabel, ylabel, legend_title=None):
    axes = ax_or_axes if isinstance(ax_or_axes, (list, tuple)) else [ax_or_axes]
    for ax in axes:
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
    axes[0].set_title(title, fontsize=11, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        axes[0].legend(title=legend_title, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def log2_sizes(ax, sizes):
    """Matrix sizes are powers of two, so a log2 x-axis spaces them evenly and keeps the
    small-size points from collapsing against the axis."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])


# ------------------------------------------------------------ experiment B ----
def plot_b(rows, outdir):
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append((int(r["size"]), num(r["speedup_vs_naive"]), num(r["gflops"])))
    sizes = sorted({int(r["size"]) for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stage in STAGE_ORDER:
        if stage not in by_stage or stage == "naive":
            continue
        pts = sorted(by_stage[stage])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                color=STAGE_COLOR.get(stage), label=stage)
    ax.axhline(1.0, color="#8c8c8c", ls="--", lw=1, label="naive (baseline)")
    log2_sizes(ax, sizes)
    finish(fig, ax, outdir, "task2_speedup_vs_size.png",
           "SGEMM speedup over naive vs matrix size",
           "matrix size (M = N = K)", "speedup over naive", "stage")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stage in STAGE_ORDER:
        if stage not in by_stage:
            continue
        pts = sorted(by_stage[stage])
        ax.plot([p[0] for p in pts], [p[2] for p in pts], marker="o", ms=5,
                color=STAGE_COLOR.get(stage), label=stage)
    log2_sizes(ax, sizes)
    finish(fig, ax, outdir, "task2_gflops_vs_size.png",
           "Achieved throughput vs matrix size",
           "matrix size (M = N = K)", "GFLOP/s", "stage")


# ------------------------------------------------------------ experiment C ----
def plot_c(rows, outdir):
    sizes = sorted({int(r["size"]) for r in rows})
    levels = [("l1d_mpki", "L1-D"), ("l2_mpki", "L2"), ("llc_mpki", "LLC")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, (key, label) in zip(axes, levels):
        by_stage = defaultdict(list)
        for r in rows:
            by_stage[r["stage"]].append((int(r["size"]), num(r[key], 0.0)))
        for stage in STAGE_ORDER:
            if stage not in by_stage:
                continue
            pts = sorted(by_stage[stage])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                    color=STAGE_COLOR.get(stage), label=stage)
        log2_sizes(ax, sizes)
        ax.set_title(f"{label} MPKI", fontsize=10)
        ax.set_xlabel("matrix size (M = N = K)")
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("misses per kilo-instruction")
    axes[0].legend(title="stage", fontsize=8)
    fig.suptitle("Cache misses per kilo-instruction vs matrix size", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "task2_mpki_vs_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------ experiment D ----
def plot_d(rows, outdir):
    for stage in sorted({r["stage"] for r in rows}):
        srows = [r for r in rows if r["stage"] == stage]
        suffix = "" if stage == "prefetch" else f"_{stage}"

        # Speedup vs distance. dist=0 is the compiled-out baseline and sits at exactly 1.0
        # by construction, so it is drawn as the reference line rather than a data point.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        by_size = defaultdict(list)
        for r in srows:
            d = int(r["dist_floats"])
            if d == 0:
                continue
            by_size[int(r["size"])].append((d, num(r["speedup_vs_noprefetch"])))
        for size in sorted(by_size):
            pts = sorted(by_size[size])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                    label=f"{size}^3")
        ax.axhline(1.0, color="#8c8c8c", ls="--", lw=1,
                   label="no SW prefetch (PREFETCH_DIST=0)")
        ax.set_xscale("log", base=2)
        dists = sorted({int(r["dist_floats"]) for r in srows if int(r["dist_floats"]) > 0})
        ax.set_xticks(dists)
        ax.set_xticklabels([str(d) for d in dists])
        finish(fig, ax, outdir, f"task2_prefetch_degree{suffix}.png",
               f"Speedup vs prefetch distance ({stage})",
               "prefetch distance (floats ahead; 16 floats = one 64 B line)",
               "speedup over no software prefetch", "matrix size")

        # The diagnostic panel: L1D MPKI against the fraction of prefetches that arrived
        # too late (their line was already resident). Together these separate "distance too
        # short" from "distance so long the line was evicted before use", which the timing
        # curve alone cannot distinguish.
        has_inef = any(num(r.get("sw_prefetch_inef"), 0) for r in srows)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        by_size = defaultdict(list)
        for r in srows:
            by_size[int(r["size"])].append(
                (int(r["dist_floats"]), num(r["l1d_mpki"], 0.0), num(r["inef_frac"], 0.0)))
        for size in sorted(by_size):
            pts = sorted(by_size[size])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                    label=f"L1-D MPKI, {size}^3")
        ax.set_xlabel("prefetch distance (floats)")
        ax.set_ylabel("L1-D MPKI")
        if has_inef:
            ax2 = ax.twinx()
            for size in sorted(by_size):
                pts = sorted(by_size[size])
                ax2.plot([p[0] for p in pts], [p[2] * 100 for p in pts], marker="^", ms=4,
                         ls=":", alpha=0.75, label=f"ineffective %, {size}^3")
            ax2.set_ylabel("prefetches issued too late (%)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Why the distance curve has that shape ({stage})", fontsize=11,
                     fontweight="bold")
        fig.tight_layout()
        path = os.path.join(outdir, f"task2_prefetch_degree_l1d{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


# ------------------------------------------------------------ experiment E ----
def plot_e(rows, outdir):
    for stage in sorted({r["stage"] for r in rows}):
        srows = [r for r in rows if r["stage"] == stage]
        suffix = "" if stage == "prefetch" else f"_{stage}"
        sizes = sorted({int(r["size"]) for r in srows})
        hints = [h for h in HINT_ORDER if any(r["hint"] == h for r in srows)]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        width = 0.8 / max(len(sizes), 1)
        xs = range(len(hints))
        for i, size in enumerate(sizes):
            vals = []
            for h in hints:
                m = [num(r["speedup_vs_noprefetch"]) for r in srows
                     if r["hint"] == h and int(r["size"]) == size]
                vals.append(m[0] if m else 0.0)
            ax.bar([x + i * width - 0.4 + width / 2 for x in xs], vals, width,
                   label=f"{size}^3")
        ax.axhline(1.0, color="#8c8c8c", ls="--", lw=1, label="no SW prefetch")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(hints)
        finish(fig, ax, outdir, f"task2_prefetch_level{suffix}.png",
               f"Speedup vs cache fill level ({stage})",
               "_mm_prefetch locality hint (both operands)",
               "speedup over no software prefetch", "matrix size")


def plot_e_grid(rows, outdir):
    for stage in sorted({r["stage"] for r in rows}):
        srows = [r for r in rows if r["stage"] == stage]
        suffix = "" if stage == "prefetch" else f"_{stage}"
        ha_list = [h for h in HINT_ORDER if any(r["hint_a"] == h for r in srows)]
        hb_list = [h for h in HINT_ORDER if any(r["hint_b"] == h for r in srows)]
        if not ha_list or not hb_list:
            continue

        grid = [[float("nan")] * len(hb_list) for _ in ha_list]
        for r in srows:
            if r["hint_a"] in ha_list and r["hint_b"] in hb_list:
                grid[ha_list.index(r["hint_a"])][hb_list.index(r["hint_b"])] = \
                    num(r["speedup_vs_noprefetch"], float("nan"))

        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(hb_list)), labels=hb_list)
        ax.set_yticks(range(len(ha_list)), labels=ha_list)
        ax.set_xlabel("hint for B (streamed, one touch per (i,j))")
        ax.set_ylabel("hint for A (reused across the j-sweep)")
        for i in range(len(ha_list)):
            for j in range(len(hb_list)):
                v = grid[i][j]
                if v == v:  # not NaN
                    ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=8,
                            color="white" if v < (max(max(r) for r in grid) * 0.75) else "black")
        fig.colorbar(im, ax=ax, label="speedup over no SW prefetch")
        size = srows[0]["size"]
        ax.set_title(f"Per-operand hint choice ({stage}, {size}^3)", fontsize=11,
                     fontweight="bold")
        fig.tight_layout()
        path = os.path.join(outdir, f"task2_prefetch_level_grid{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")


# ------------------------------------------------------------ experiment F ----
def plot_f(rows, outdir):
    stages = [s for s in STAGE_ORDER if any(r["stage"] == s for r in rows)]
    sizes = sorted({int(r["size"]) for r in rows})

    fig, axes = plt.subplots(1, len(stages), figsize=(4.6 * len(stages), 4.4), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        cells = {("on", "on"): [], ("on", "off"): [], ("off", "on"): [], ("off", "off"): []}
        for r in rows:
            if r["stage"] != stage:
                continue
            t = num(r["time_ms"])
            if t:
                cells[(r["hwpf"], r["swpf"])].append((int(r["size"]), t))

        # Normalise every bar against the same reference: hardware prefetch on, software
        # prefetch off. That is the configuration a normal build runs in, so each bar reads
        # as "what this change bought relative to shipping as-is".
        ref = dict(cells[("on", "off")])
        width = 0.2
        xs = range(len(sizes))
        for i, (hw, sw) in enumerate([("on", "off"), ("on", "on"), ("off", "off"), ("off", "on")]):
            d = dict(cells[(hw, sw)])
            vals = [(ref[s] / d[s]) if (s in d and s in ref and d[s]) else 0.0 for s in sizes]
            ax.bar([x + i * width - 0.4 + width / 2 for x in xs], vals, width,
                   label=f"HW {hw}, SW {sw}")
        ax.axhline(1.0, color="#8c8c8c", ls="--", lw=1)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_title(stage, fontsize=10)
        ax.set_xlabel("matrix size")
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("speedup vs (HW on, SW off)")
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle("Hardware prefetcher vs software prefetcher", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "task2_hw_prefetcher.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------ experiment G ----
def plot_g(time_rows, instr_rows, outdir):
    if time_rows:
        sizes = sorted({int(r["size"]) for r in time_rows})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for v in VARIANT_ORDER:
            pts = sorted((int(r["size"]), num(r["speedup"]))
                         for r in time_rows if r["variant"] == v and num(r["speedup"]))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                    label=VARIANT_LABEL.get(v, v))
        ax.axhline(1.0, color="#8c8c8c", ls="--", lw=1)
        log2_sizes(ax, sizes)
        finish(fig, ax, outdir, "task2_simd_width_speedup.png",
               "Speedup over naive vs SIMD register width",
               "matrix size (M = N = K)", "speedup over naive", "SIMD width")

    if instr_rows:
        sizes = sorted({int(r["size"]) for r in instr_rows})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for v in VARIANT_ORDER:
            pts = sorted((int(r["size"]), num(r["instructions_per_rep"]))
                         for r in instr_rows
                         if r["variant"] == v and num(r["instructions_per_rep"]))
            if not pts:
                continue
            plotted = True
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                    label=VARIANT_LABEL.get(v, v))
        if plotted:
            ax.set_yscale("log")
            log2_sizes(ax, sizes)
            finish(fig, ax, outdir, "task2_simd_width_instructions.png",
                   "Retired instructions per SGEMM vs SIMD register width",
                   "matrix size (M = N = K)", "instructions per call (log scale)",
                   "SIMD width")


# ------------------------------------------------------------ experiment H ----
def plot_h(rows, outdir):
    good = [r for r in rows if r.get("correct") == "yes" and num(r["time_ms"])]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    by_bk = defaultdict(list)
    for r in good:
        by_bk[int(r["block_k"])].append(
            (num(r["working_set_kib"]), num(r["gflops"]), f"{r['block_m']}x{r['block_n']}"))
    for bk in sorted(by_bk):
        pts = sorted(by_bk[bk])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, ls="",
                label=f"BLOCK_K={bk}")
    ax.set_xscale("log", base=2)

    # Mark the cache capacities: a configuration's performance should fall off once its
    # block working set stops fitting.
    for cap, label in _cache_caps():
        ax.axvline(cap, color="#c0392b", ls="--", lw=1, alpha=0.7)
        ax.text(cap, ax.get_ylim()[1] * 0.98, f" {label}", fontsize=7.5, color="#c0392b",
                va="top", rotation=90)
    finish(fig, ax, outdir, "task2_block_size.png",
           "Throughput vs block working-set size",
           "block working set (KiB, log scale)", "GFLOP/s", "K-block")


def _cache_caps():
    caps = []
    try:
        import subprocess
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("L1d cache:") or line.startswith("L2 cache:"):
                parts = line.split(":", 1)[1].strip()
                val = float(parts.split()[0])
                unit = parts.split()[1]
                kib = val * (1024 if unit.startswith("Mi") else 1)
                inst = 1
                if "(" in parts and "instance" in parts:
                    inst = int(parts.split("(")[1].split()[0])
                caps.append((kib / inst, "L1d" if "L1d" in line else "L2"))
    except Exception:
        pass
    return caps


# ------------------------------------------------------------ experiment I ----
def plot_i(rows, outdir):
    def median(vals):
        vals = sorted(v for v in vals if v is not None)
        return vals[len(vals) // 2] if vals else 0.0

    builds = ["stock", "student"]
    prompt = [median([num(r["prompt_eval_tps"]) for r in rows if r["build"] == b])
              for b in builds]
    gen = [median([num(r["gen_eval_tps"]) for r in rows if r["build"] == b]) for b in builds]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    xs = range(2)
    ax.bar([x - 0.2 for x in xs], prompt, 0.4, label="prompt eval (batched, fast path)")
    ax.bar([x + 0.2 for x in xs], gen, 0.4, label="generation (M=1, scalar tail)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["llama.cpp native", "matmul_optimized"])
    finish(fig, ax, outdir, "task2_llama_tokens_per_sec.png",
           "llama.cpp inference: native SGEMM vs matmul_optimized",
           "", "tokens per second")


# ------------------------------------------------------------------ main ------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "results"))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "..", "plots"))
    args = ap.parse_args()

    res, out = os.path.abspath(args.results), os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)
    print(f"reading  {res}")
    print(f"writing  {out}")

    jobs = [
        ("B  speedup vs size", ["b_speedup_vs_size.csv"], plot_b),
        ("C  cache metrics", ["c_cache_metrics.csv"], plot_c),
        ("D  prefetch distance", ["d_prefetch_distance.csv"], plot_d),
        ("E  prefetch level", ["e_prefetch_level.csv"], plot_e),
        ("E  prefetch level grid", ["e_prefetch_level_grid.csv"], plot_e_grid),
        ("F  hardware prefetcher", ["f_hw_prefetcher.csv"], plot_f),
        ("G  SIMD width", ["g_simd_width_time.csv", "g_simd_width_instructions.csv"], plot_g),
        ("H  block size", ["h_block_size.csv"], plot_h),
        ("I  llama.cpp", ["i_llama.csv"], plot_i),
    ]

    any_done = False
    for label, files, fn in jobs:
        data = [read(os.path.join(res, f)) for f in files]
        if all(d is None for d in data):
            print(f"skip {label}: no data yet")
            continue
        print(f"{label}:")
        try:
            fn(*data, out)
            any_done = True
        except Exception as e:  # one broken CSV should not block the other figures
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    if not any_done:
        print("\nnothing plotted. Run the exp_*.sh scripts first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
