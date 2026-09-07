import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "plots"))
os.makedirs(OUT, exist_ok=True)
d = json.load(open(os.path.join(HERE, "results.json")))
SIZES = d["grid_sizes"]
KS = d["grid_ks"]
KREF = "3"
NREF = "1024"

def ms(st, N):   return d["grid"][st][str(N)][KREF]
def msk(st, K):  return d["grid"][st][NREF][str(K)]
def mpki(st, N): return d["baseline"][st][str(N)]["mpki"]

plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

# 1. task 1d: speedup vs matrix size, all techniques
LAB = [("reorder", "Loop reordering"), ("unroll", "Loop unrolling"),
       ("tile", "Tiling"), ("simd_nt", "SIMD (AVX2)"),
       ("simd", "Tiling + SIMD"), ("optimized", "All combined")]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for st, lab in LAB:
    ax.plot(SIZES, [ms("naive", N) / ms(st, N) for N in SIZES],
            marker="o", label=lab)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
ax.set_xticklabels([str(s) for s in SIZES])
ax.set_xlabel("matrix size N (N x N, K=3)")
ax.set_ylabel("speedup over naive")
ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
ax.set_title("Task 1D: speedup by optimization technique vs matrix size")
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/task1d_speedup_lines.png"); plt.close(fig)

# 1b. same data as grouped bars, the format asked for in figure 1.1
import numpy as np
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(SIZES))
w = 0.8 / len(LAB)
for i, (st, lab) in enumerate(LAB):
    vals = [ms("naive", N) / ms(st, N) for N in SIZES]
    ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=lab)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in SIZES])
ax.set_xlabel("matrix size N (N x N, K=3)")
ax.set_ylabel("speedup over naive (normalized)")
ax.set_yscale("log"); ax.set_ylim(0.5, 90)
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
ax.grid(axis="x", visible=False)
ax.set_title("Task 1D: speedup by optimization technique vs matrix size")
ax.legend(fontsize=8, ncol=3, loc="upper center")
fig.tight_layout(); fig.savefig(f"{OUT}/task1_speedup_vs_size.png"); plt.close(fig)

# 2. speedup vs kernel size
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for st, lab in LAB:
    ax.plot(KS, [msk("naive", K) / msk(st, K) for K in KS],
            marker="o", label=lab)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xticks(KS)
ax.set_xlabel("kernel size K (K x K, N=1024)")
ax.set_ylabel("speedup over naive")
ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
ax.set_title("Task 1D: speedup by optimization technique vs kernel size")
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/task1_speedup_vs_kernel.png"); plt.close(fig)

# 3. task 1b: l1-d mpki vs matrix size, per tile size
fig, ax = plt.subplots(figsize=(6.4, 4.0))
TS = d["sizes"]
for key in d["tile_sweep"]:
    ax.plot(TS, [d["tile_sweep"][key][str(N)]["mpki"] for N in TS],
            marker="o", ms=3, label=f"tile {key}")
ax.plot(TS, [mpki("naive", N) for N in TS], marker="s", color="k",
        lw=2, label="naive")
ax.set_xscale("log", base=2); ax.set_xticks(TS)
ax.set_xticklabels([str(s) for s in TS])
ax.set_xlabel("matrix size N (N x N, K=3)")
ax.set_ylabel("L1-D MPKI")
ax.set_title("Task 1B: L1-D MPKI vs matrix size for different tile sizes")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/task1b_mpki_vs_size.png"); plt.close(fig)

# 4. task 1b: speedup vs matrix size, per tile size
fig, ax = plt.subplots(figsize=(6.4, 4.0))
TS = d["sizes"]
for key in d["tile_sweep"]:
    ax.plot(TS, [d["baseline"]["naive"][str(N)]["ms"]
                 / d["tile_sweep"][key][str(N)]["ms"]
                 for N in TS], marker="o", ms=3, label=f"tile {key}")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xscale("log", base=2); ax.set_xticks(TS)
ax.set_xticklabels([str(s) for s in TS])
ax.set_xlabel("matrix size N (N x N, K=3)")
ax.set_ylabel("speedup over naive")
ax.set_title("Task 1B: tiled speedup vs matrix size for different tile sizes")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/task1b_speedup_vs_size.png"); plt.close(fig)

# 5. task 1c: simd width
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
for st, lab in [("simd128", "128-bit (SSE)"), ("simd", "256-bit (AVX2)")]:
    axes[0].plot(SIZES, [ms("naive", N) / ms(st, N) for N in SIZES],
                 marker="o", label=lab)
axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
axes[0].set_xscale("log", base=2); axes[0].set_xticks(SIZES)
axes[0].set_xticklabels([str(s) for s in SIZES])
axes[0].set_xlabel("matrix size N (K=3)")
axes[0].set_ylabel("speedup over naive")
axes[0].set_title("Speedup vs matrix size")
axes[0].legend(fontsize=8)
w = [0, 1]
ins = [d["baseline"][s]["2048"]["instructions"] / 1e9
       for s in ["simd128", "simd"]]
nai = d["baseline"]["naive"]["2048"]["instructions"] / 1e9
axes[1].bar([-1] + w, [nai] + ins, color=["0.5", "tab:orange", "tab:green"])
axes[1].set_xticks([-1] + w)
axes[1].set_xticklabels(["naive", "128-bit", "256-bit"])
axes[1].set_ylabel("instructions retired (x1e9)")
axes[1].set_title("Instruction count at 2048x2048, K=3")
fig.suptitle("Task 1C: SIMD width comparison (no AVX-512 on this CPU)",
             fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/task1c_simd_width.png"); plt.close(fig)
print("plots written to", OUT)

# 6. task 1b: l2 misses vs tile footprint at n=2048
foot, l2, labs = [], [], []
for key in d["tile_sweep"]:
    th, tw = map(int, key.split("x"))
    kb = (th*tw*4 + (th+2)*(tw+2)*4) / 1024.0
    foot.append(kb)
    l2.append(d["tile_sweep"][key]["2048"]["l2_miss"] / 4.0 * 64 / 2**20)
    labs.append(key)
order = sorted(range(len(foot)), key=lambda i: foot[i])
foot = [foot[i] for i in order]
l2 = [l2[i] for i in order]
labs = [labs[i] for i in order]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(foot, l2, marker="o", color="tab:blue", label="tiled")
for x, y, t in zip(foot, l2, labs):
    ax.annotate(t, (x, y), fontsize=7, xytext=(0, 6),
                textcoords="offset points", ha="center")
unt = d["baseline"]["reorder"]["2048"]["l2_miss"] / 4.0 * 64 / 2**20
ax.axhline(unt, color="tab:red", ls="--", lw=1,
           label=f"untiled reorder ({unt:.0f} MiB)")
ax.axhline(32, color="k", ls=":", lw=1, label="compulsory 32 MiB")
ax.axvline(1280, color="gray", ls="-.", lw=1, label="L2 capacity (1.25 MiB)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("tile footprint: output tile + input halo (KiB)")
ax.set_ylabel("L2 miss traffic per run (MiB)")
ax.set_title("Task 1B: L2 traffic vs tile footprint (2048x2048, K=3)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/task1b_l2_vs_tile.png"); plt.close(fig)
print("l2 plot done")
