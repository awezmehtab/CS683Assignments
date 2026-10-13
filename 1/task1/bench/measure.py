import json, re, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = "2"

def build(th=64, tw=256):
    subprocess.run(["make", "-B", f"TILE_H={th}", f"TILE_W={tw}"],
                   cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_time(stage, H, W, K, reps=7):
    out = subprocess.run(["taskset", "-c", CORE, "./bench", stage,
                          str(H), str(W), str(K), str(reps)],
                         cwd=HERE, capture_output=True, text=True, check=True)
    return float(out.stdout.strip().split("\n")[-1])

EVENTS = "instructions,L1-dcache-loads,L1-dcache-load-misses"

def run_perf(stage, H, W, K, reps=3):
    out = subprocess.run(["perf", "stat", "-x,", "-e", EVENTS,
                          "taskset", "-c", CORE, "./bench", stage,
                          str(H), str(W), str(K), str(reps)],
                         cwd=HERE, capture_output=True, text=True)
    vals = {}
    for line in out.stderr.split("\n"):
        f = line.split(",")
        if len(f) < 3 or not f[0].isdigit():
            continue
        name = f[2]
        if name.startswith("cpu_core/"):
            vals[name[len("cpu_core/"):-3]] = int(f[0])
    ins = vals.get("instructions", 0)
    mis = vals.get("L1-dcache-load-misses", 0)
    lds = vals.get("L1-dcache-loads", 0)
    return {"instructions": ins, "l1_loads": lds, "l1_misses": mis,
            "mpki": 1000.0 * mis / ins if ins else 0.0}

SIZES = [256, 512, 1024, 2048]
TILES = [(16, 64), (32, 128), (64, 64), (64, 128),
         (64, 256), (128, 256), (128, 512), (256, 512)]

data = {"sizes": SIZES, "tiles": [list(t) for t in TILES]}

# baseline stages at default tile geometry
build()
data["baseline"] = {}
for st in ["naive", "reorder", "unroll", "tile", "simd", "simd128",
           "optimized"]:
    data["baseline"][st] = {}
    for N in SIZES:
        t = run_time(st, N, N, 3)
        p = run_perf(st, N, N, 3)
        data["baseline"][st][N] = {"ms": t, **p}
        print(f"[base] {st:10s} N={N:5d} {t:9.3f} ms  MPKI={p['mpki']:7.3f}",
              flush=True)

# kernel-size sweep
KS = [3, 5, 7, 9]
data["kernel"] = {}
for st in ["naive", "reorder", "unroll", "tile", "simd", "simd128",
           "optimized"]:
    data["kernel"][st] = {}
    for K in KS:
        t = run_time(st, 1024, 1024, K)
        data["kernel"][st][K] = t
        print(f"[kern] {st:10s} K={K} {t:9.3f} ms", flush=True)
data["kernel_sizes"] = KS

# tile-size sweep
data["tile_sweep"] = {}
for (th, tw) in TILES:
    build(th, tw)
    key = f"{th}x{tw}"
    data["tile_sweep"][key] = {}
    for N in SIZES:
        t = run_time("tile", N, N, 3)
        p = run_perf("tile", N, N, 3)
        data["tile_sweep"][key][N] = {"ms": t, **p}
        print(f"[tile] {key:9s} N={N:5d} {t:9.3f} ms  MPKI={p['mpki']:7.3f}",
              flush=True)

build()
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(data, f, indent=1)
