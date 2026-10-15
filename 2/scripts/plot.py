#!/usr/bin/env python3
"""grouped bar charts from parse.py's CSV, one png per metric
usage: scripts/plot.py <csv> <out_prefix> <config>[=Name],... <metric>...
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# speedup y axis is zoomed to the data; ipc and speedup bars get values
ZOOM = {"speedup"}
VALUES = {"ipc", "speedup"}

LABELS = {
    "ipc": "IPC",
    "speedup": "Speedup over baseline",
    "l1d_load_mpki": "L1D load MPKI",
    "l1d_pf_accuracy": "L1D prefetch accuracy (%)",
    "l1d_coverage": "L1D coverage (%)",
    "l2c_load_mpki": "L2C load MPKI",
    "l2c_instr_load_mpki": "L2C instruction load MPKI",
    "l2c_data_load_mpki": "L2C data load MPKI",
}


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    csv_path, prefix, cfg_arg = sys.argv[1:4]
    metrics = sys.argv[4:]

    configs = []
    for item in cfg_arg.split(","):
        key, _, name = item.partition("=")
        configs.append((key, name or key))

    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data[(row["config"], row["trace"])] = row
    traces = sorted({t for (c, t) in data if c in dict(configs)})

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    width = 0.8 / len(configs)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(traces)), 4.5))
        allvals = []
        for i, (key, name) in enumerate(configs):
            vals = [data.get((key, t), {}).get(metric, "") for t in traces]
            vals = [float(v) if v not in ("", None) else 0 for v in vals]
            allvals += vals
            xs = [x + (i - (len(configs) - 1) / 2) * width
                  for x in range(len(traces))]
            bars = ax.bar(xs, vals, width, label=name, edgecolor="grey")
            if metric in VALUES:
                ax.bar_label(bars, fmt="%.3f", fontsize=6, rotation=90,
                             padding=2)
        if metric in ZOOM and allvals:
            lo, hi = min(allvals), max(allvals)
            pad = max((hi - lo) * 0.4, hi * 0.005)
            ax.set_ylim(lo - pad, hi + 2 * pad)
        elif allvals:
            ax.set_ylim(min(0, min(allvals)), max(allvals) * 1.3)
        if metric == "speedup":
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(traces)))
        ax.set_xticklabels(traces)
        ax.set_ylabel(LABELS.get(metric, metric))
        ax.legend(fontsize=7, ncol=min(3, len(configs)))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_{metric}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {prefix}_{metric}.png")


if __name__ == "__main__":
    main()
