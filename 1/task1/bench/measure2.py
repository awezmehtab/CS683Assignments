import json, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = "2"
EVENTS = "instructions,L1-dcache-loads,L1-dcache-load-misses"

def build(th=64, tw=256):
    subprocess.run(["make", "-B", f"TILE_H={th}", f"TILE_W={tw}"], cwd=HERE,
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

def run_time(stage, H, W, K, reps=7):
    o = subprocess.run(["taskset", "-c", CORE, "./bench", stage, str(H), str(W),
                        str(K), str(reps)], cwd=HERE, capture_output=True,
                       text=True, check=True)
    return float(o.stdout.strip().split("\n")[-1])

def run_perf(stage, H, W, K, reps=3):
    o = subprocess.run(["perf", "stat", "-x,", "-e", EVENTS, "taskset", "-c", CORE,
                        "./bench", stage, str(H), str(W), str(K), str(reps)],
                       cwd=HERE, capture_output=True, text=True)
    v = {}
    for line in o.stderr.split("\n"):
        f = line.split(",")
        if len(f) < 3 or not f[0].isdigit():
            continue
        if f[2].startswith("cpu_core/"):
            v[f[2][len("cpu_core/"):-3]] = int(f[0])
    ins = v.get("instructions", 0)
    return {"instructions": ins, "l1_loads": v.get("L1-dcache-loads", 0),
            "l1_misses": v.get("L1-dcache-load-misses", 0),
            "mpki": 1000.0*v.get("L1-dcache-load-misses", 0)/ins if ins else 0}

build()
d = json.load(open(os.path.join(HERE, "results.json")))
d["baseline"]["simd_nt"] = {}
for N in d["sizes"]:
    t = run_time("simd_nt", N, N, 3)
    p = run_perf("simd_nt", N, N, 3)
    d["baseline"]["simd_nt"][str(N)] = {"ms": t, **p}
    print(f"[base] simd_nt N={N:5d} {t:9.3f} ms MPKI={p['mpki']:7.3f}")
d["kernel"]["simd_nt"] = {}
for K in d["kernel_sizes"]:
    t = run_time("simd_nt", 1024, 1024, K)
    d["kernel"]["simd_nt"][str(K)] = t
    print(f"[kern] simd_nt K={K} {t:9.3f} ms")
json.dump(d, open(os.path.join(HERE, "results.json"), "w"), indent=1)
