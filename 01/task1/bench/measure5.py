import json, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = "2"
STAGES = ["naive", "reorder", "unroll", "tile", "simd_nt", "simd", "simd128", "optimized"]
SIZES = [128, 256, 512, 1024, 2048]
KS = [3, 5, 7, 9]

def t(stage, H, W, K, reps=7):
    o = subprocess.run(["taskset", "-c", CORE, "./bench", stage, str(H), str(W),
        str(K), str(reps)], cwd=HERE, capture_output=True, text=True, check=True)
    return float(o.stdout.strip().split("\n")[-1])
# interleave stages within each (N,K) so drift hits all stages equally
grid = {s: {} for s in STAGES}
for N in SIZES:
    for K in KS:
        for s in STAGES: grid[s].setdefault(str(N), {})
        for rnd in range(3):
            for s in STAGES:
                v = t(s, N, N, K)
                cur = grid[s][str(N)].get(str(K))
                grid[s][str(N)][str(K)] = v if cur is None else min(cur, v)
        print(f"N={N:5d} K={K} " + " ".join(
            f"{s}={grid[s][str(N)][str(K)]:.3f}" for s in STAGES), flush=True)
d = json.load(open(os.path.join(HERE, "results.json")))
d["grid"] = grid
d["grid_sizes"] = SIZES
d["grid_ks"] = KS
json.dump(d, open(os.path.join(HERE, "results.json"), "w"), indent=1)
