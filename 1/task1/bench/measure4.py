import json, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = "2"
EV = "instructions,l2_rqsts.miss,LLC-load-misses"

def build(th=64, tw=256):
    subprocess.run(["make", "-B", f"TILE_H={th}", f"TILE_W={tw}"], cwd=HERE,
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def perf(stage, H, W, K, reps=3):
    o = subprocess.run(["perf", "stat", "-x,", "-e", EV, "taskset", "-c", CORE,
        "./bench", stage, str(H), str(W), str(K), str(reps)], cwd=HERE,
        capture_output=True, text=True)
    v = {}
    for ln in o.stderr.split("\n"):
        f = ln.split(",")
        if len(f)<3 or not f[0].isdigit(): continue
        if f[2].startswith("cpu_core/"): v[f[2].split("/")[1]] = int(f[0])
    ins = v.get("instructions", 1)
    return {"l2_miss":v.get("l2_rqsts.miss", 0),
            "llc_miss":v.get("LLC-load-misses", 0),
            "l2_mpki":1000.0*v.get("l2_rqsts.miss", 0)/ins,
            "llc_mpki":1000.0*v.get("LLC-load-misses", 0)/ins,
            "reps":reps}
d = json.load(open(os.path.join(HERE, "results.json")))
build()
for st in ["naive", "reorder", "unroll", "tile", "simd", "simd128", "simd_nt",
           "optimized"]:
    for N in d["sizes"]:
        p = perf(st, N, N, 3); d["baseline"][st][str(N)].update(p)
        print(f"[l2] {st:10s} N={N:5d} L2miss={p['l2_miss']:12,} "
              f"L3miss={p['llc_miss']:12,}", flush=True)
for (th, tw) in d["tiles"]:
    build(th, tw); key = f"{th}x{tw}"
    for N in d["sizes"]:
        p = perf("tile", N, N, 3); d["tile_sweep"][key][str(N)].update(p)
        print(f"[l2] tile {key:9s} N={N:5d} L2miss={p['l2_miss']:12,}",
              flush=True)
build()
json.dump(d, open(os.path.join(HERE, "results.json"), "w"), indent=1)
