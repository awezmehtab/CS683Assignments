#!/usr/bin/env python3
"""results/<label>/<trace>.txt -> CSV (speedup, coverage vs baseline)
usage: scripts/parse.py [results_dir] [baseline_label] > results/summary.csv
"""
import csv
import re
import sys
from pathlib import Path

NUM = r"([-+0-9.eE]+|-?nan|-?inf)"
PATTERNS = {
    "ipc": rf"^CPU 0 cumulative IPC: {NUM}",
    "instructions": rf"^CPU 0 cumulative IPC: \S+ instructions: (\d+)",
    "l1d_load_miss": r"^L1D LOAD\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)",
    "l1d_load_mpki": rf"^L1D LOAD\s+ACCESS:.*MPKI:\s+{NUM}",
    "l1d_pf_useful": r"^L1D USEFUL LOAD PREFETCHES:\s+(\d+)",
    "l1d_pf_issued": r"^L1D USEFUL LOAD PREFETCHES:.*LOWER LEVEL:\s+(\d+)",
    "l2c_load_miss": r"^L2C LOAD\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)",
    "l2c_load_mpki": rf"^L2C LOAD\s+ACCESS:.*MPKI:\s+{NUM}",
    "l2c_instr_load_mpki": rf"^L2C INSTRUCTION LOAD MPKI: {NUM}",
    "l2c_data_load_mpki": rf"^L2C DATA LOAD MPKI: {NUM}",
}
PATTERNS = {k: re.compile(v) for k, v in PATTERNS.items()}


def parse(path):
    text = path.read_text(errors="replace")
    roi = text.split("Region of Interest Statistics", 1)
    if len(roi) != 2:
        return None
    row = {}
    for line in roi[1].splitlines():
        line = line.strip()
        for key, pat in PATTERNS.items():
            if key not in row:
                m = pat.match(line)
                if m:
                    row[key] = float(m.group(1))
    return row


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    base = sys.argv[2] if len(sys.argv) > 2 else "baseline"

    rows = {}
    for f in sorted(results.glob("*/*.txt")):
        row = parse(f)
        if row is None:
            print(f"skipping unfinished {f}", file=sys.stderr)
            continue
        rows[(f.parent.name, f.stem)] = row

    out = csv.writer(sys.stdout)
    cols = list(PATTERNS) + ["speedup", "l1d_pf_accuracy", "l1d_coverage"]
    out.writerow(["config", "trace"] + cols)
    for (cfg, trace), r in sorted(rows.items()):
        b = rows.get((base, trace))
        issued = r.get("l1d_pf_issued", 0)
        r["l1d_pf_accuracy"] = (
            100 * r.get("l1d_pf_useful", 0) / issued if issued else "")
        r["speedup"] = r["ipc"] / b["ipc"] if b else ""
        r["l1d_coverage"] = (
            100 * (b["l1d_load_miss"] - r["l1d_load_miss"])
            / b["l1d_load_miss"]
            if b and b.get("l1d_load_miss") else "")
        out.writerow([cfg, trace] + [
            "" if r.get(c, "") == "" else f"{r[c]:.6g}" for c in cols])


if __name__ == "__main__":
    main()
