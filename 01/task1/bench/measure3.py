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
        if len(f)<3 or not f[0].isdigit(): continue
        if f[2].startswith("cpu_core/"):
            v[f[2].split("/")[1]] = int(f[0])
    ins = v.get("instructions", 0); mis = v.get("L1-dcache-load-misses", 0)
    return {"instructions":ins, "l1_loads":v.get("L1-dcache-loads", 0),
            "l1_misses":mis, "mpki":1000.0*mis/ins if ins else 0.0,
            "miss_rate":100.0*mis/v["L1-dcache-loads"]
                        if v.get("L1-dcache-loads") else 0.0}

d = json.load(open(os.path.join(HERE, "results.json")))
STAGES = ["naive", "reorder", "unroll", "tile", "simd", "simd128", "optimized"]

build()
# simd_nt: new stage, needs times too
d["baseline"]["simd_nt"] = {}
for N in d["sizes"]:
    d["baseline"]["simd_nt"][str(N)] = {"ms":run_time("simd_nt", N, N, 3)}
d["kernel"]["simd_nt"] = {str(K):run_time("simd_nt", 1024, 1024, K)
                        for K in d["kernel_sizes"]}
print("simd_nt timed", flush=True)

for st in STAGES+["simd_nt"]:
    for N in d["sizes"]:
        p = run_perf(st, N, N, 3)
        d["baseline"][st][str(N)].update(p)
        print(f"[base] {st:10s} N={N:5d} MPKI={p['mpki']:8.3f} "
              f"missrate={p['miss_rate']:6.2f}% ins={p['instructions']:,}",
              flush=True)

for (th, tw) in d["tiles"]:
    build(th, tw); key = f"{th}x{tw}"
    for N in d["sizes"]:
        p = run_perf("tile", N, N, 3)
        d["tile_sweep"][key][str(N)].update(p)
        print(f"[tile] {key:9s} N={N:5d} MPKI={p['mpki']:8.3f}", flush=True)

build()
json.dump(d, open(os.path.join(HERE, "results.json"), "w"), indent=1)
