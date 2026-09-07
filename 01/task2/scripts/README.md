# Task 2 experiment scripts

One script per experiment, each writing a CSV into `results/`. `plot_all.py` turns those
into the report figures in `../../plots/`.

```bash
./run_all.sh              # everything runnable here, in dependency order, then plots
sudo -E ./run_all.sh      # including F, which needs root
QUICK=1 ./run_all.sh      # small sizes, for checking the pipeline works
```

Or one at a time (this is the intended order, since D's answer feeds E and F):

| Script | Deliverable | Needs | Output |
|---|---|---|---|
| `exp_a_env.sh` | prerequisite | perf | `results/env.txt` |
| `exp_b_speedup_vs_size.sh` | 2A.1, 2A.3, 2C | — | `b_speedup_vs_size.csv` |
| `exp_c_cache_metrics.sh` | 2A.2 | perf | `c_cache_metrics.csv` |
| `exp_d_prefetch_distance.sh` | 2A.4, 2C | perf | `d_prefetch_distance.csv` |
| `exp_e_prefetch_level.sh` | 2A.4, 2C | perf | `e_prefetch_level{,_grid}.csv` |
| `exp_f_hw_prefetcher.sh` | 2A.5 | **root** + `msr-tools` | `f_hw_prefetcher.csv` |
| `exp_g_simd_width.sh` | 2B (all) | perf | `g_simd_width_*.csv` |
| `exp_h_block_size.sh` | tuning | — | `h_block_size.csv` |
| `exp_i_llama.sh` | README §4 | network | `i_llama.{csv,log}` |

## Setup on the measurement machine

```bash
sudo sysctl kernel.perf_event_paranoid=1     # required by A, C, D, E, G
sudo cpupower frequency-set -g performance   # otherwise frequency ramp becomes "variance"
sudo apt install msr-tools && sudo modprobe msr   # F only
```

Run `exp_a_env.sh` first. It prints the LOGICAL -> ACTUAL perf event mapping and fails
loudly if perf cannot open events, instead of letting later scripts emit silent zero
columns.

## Common knobs

Every script reads these from the environment:

```
CORE=2                  core to pin all measurements to
REPS=3  WARMUP=1        timed / untimed repetitions
SIZES="128 ... 2048"    matrix sizes (M=N=K)
STAGES="simd prefetch optimized"
PF_DIST=64              prefetch distance held fixed by E and F
RESULTS_DIR= PLOTS_DIR=
```

## Two things worth knowing before reading the numbers

**The speedup denominator.** The assignment defines speedup as *time without software
prefetching / time with software prefetching*. That requires the same kernel with the
prefetches removed, so `PREFETCH_DIST=0` compiles every `_mm_prefetch` away via the `PF()`
macro in `matmul_prefetch.cpp` and `matmul_optimized.cpp`. Dividing by `matmul_simd`
instead would report the cache-blocking win as though it were a prefetching win, because
`matmul_simd` is un-blocked. D, E and F all assert (against the disassembly, via
`assert_noprefetch`) that the baseline binary really contains zero prefetch instructions
before using it as a denominator.

**Every knob needs its `#ifndef` guard.** `BLOCK_M/N/K`, `PREFETCH_DIST` and
`PREFETCH_HINT_A/B` must each be written as

```cpp
#ifndef BLOCK_M
#define BLOCK_M 32
#endif
```

Without the guard, the source's own `#define` *redefines* the `-D` value from the command
line, so `-DBLOCK_M=64` compiles as 32. The sweep still runs, still emits a CSV, and still
draws a plot, but every row measures the identical binary while being labelled as a
different configuration. It produces no error and no visibly wrong number. If you edit
those files, keep the guards.

## Counters

The PDF names Intel event names. On AMD they do not exist, and the generic kernel aliases
(`L1-dcache-load-misses`) are unmapped on several AMD parts. `common.sh` addresses counters
by logical name and resolves each by probing candidates in order, so the same scripts work
on either vendor:

| Logical | AMD Zen 3 | Intel |
|---|---|---|
| `l1d_miss` | `ls_mab_alloc.load_store_allocations` | `MEM_LOAD_RETIRED.L1_MISS` |
| `l2_miss` | `l2_cache_misses_from_dc_misses` | `l2_rqsts.miss` |
| `llc_miss` | `ls_any_fills_from_sys.mem_io_local` | `LONGEST_LAT_CACHE.MISS` |
| `sw_prefetch` | `ls_pref_instr_disp.prefetch` | `sw_prefetch_access.*` |
| `sw_prefetch_inef` | `ls_inef_sw_pref.data_pipe_sw_pf_dc_hit` | (none) |
| `hwpf` | `l2_cache_accesses_from_l2_hwpf` | `l2_rqsts.all_pf` |

`exp_a_env.sh` prints the mapping actually used; quote it in the report.

Counts come from `perf_delta()`, which runs each binary twice, once doing `$REPS` kernel
repetitions and once with `--reps=0` (setup only), and subtracts. Without that, the page
faults from first-touching the working set and the RNG fill sit inside every count and
dominate at the small sizes.

## Adding a counter

Add candidate names to `_candidates()` in `common.sh`, most specific first, and add the
logical name to the two lists in `resolve_all_events()` / `print_event_map()`. Then pass it
to `perf_delta` and read it back from `PD[<logical>]`.
