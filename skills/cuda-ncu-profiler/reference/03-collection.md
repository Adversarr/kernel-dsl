# Profile Collection Commands

This document lists the exact `ncu` commands to run for Python DSL profile drivers and generated CUDA kernels.

---

## Prerequisites recap

- `ncu` available on `PATH`
- writable `HOME`
- a Python profile driver under `$PROFILE_RUN_DIR/target/`
- a kernel token chosen from the Python-level symbol name or another stable identifier
- a validated selector, usually `regex:.*<kernel_token>.*`

Correctness is assumed to be settled before this phase begins.

Quick permission test:

```bash
ncu --section SpeedOfLight -k "regex:.*YOUR_KERNEL_TOKEN.*" -c 1 \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

If you see `ERR_NVGPUCTRPERM`, see [`09-common-issues.md`](09-common-issues.md).

---

## Discover the regex token first

The default discovery path for DSL-generated kernels is:

1. start from the Python entrypoint name, for example `some_kernel`
2. build `regex:.*some_kernel.*`
3. run a cheap inspection or first collection pass
4. confirm the captured demangled name in the resulting report

Use emitted/generated CUDA source or runtime debug output only if the obvious Python token is not specific enough.

---

## Recipe 1: Full overview (first pass)

Collects all standard sections plus PM sampling. This is the mandatory first run.

```bash
ncu --set full \
    --section PmSampling \
    --section PmSampling_WarpStates \
    -k "regex:.*KERNEL_TOKEN.*" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

| Flag | Meaning |
|---|---|
| `--set full` | Run all built-in sections |
| `--section PmSampling` | Add performance-monitor time-series data |
| `--section PmSampling_WarpStates` | Time-series of warp stall states |
| `-k "regex:..."` | Only profile kernels whose demangled name matches |
| `-c 1` | Only profile one matching launch |
| `-o ...` | Output path; `.ncu-rep` is appended automatically |

Replay count is typically 45-50 passes.

---

## Recipe 2: Source-level profile (second pass)

Collects per-PC stall sampling data. This only works when the target path preserves usable source mapping.

```bash
ncu --set source \
    --section SourceCounters \
    -k "regex:.*KERNEL_TOKEN.*" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/source_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

Use this report with `extract_stall_hotspots.py`.

---

## Recipe 3: Warmup before collection

If the first launch triggers JIT or compile work, run the profile driver once without NCU first:

```bash
python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" --warmup-only [args]
```

Then collect with NCU against the steady-state launch path. If the driver cannot separate warmup from steady-state, use `-s` and `-c` to target the right launch.

---

## Recipe 4: Skip compile-only or helper launches

If the profile driver launches matching kernels multiple times:

```bash
ncu --set full \
    -k "regex:.*KERNEL_TOKEN.*" \
    -s 1 -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

- `-s N` skips the first `N` matching launches
- `-c N` limits collection to `N` matching launches

This is the main way to avoid compile-only or warmup launches polluting the report.

---

## Recipe 5: Details page (quick rule summary)

No need to collect again:

```bash
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page details \
    > "$PROFILE_RUN_DIR/analysis/details_<tag>.txt"
```

Always read `details_<tag>.txt` early. NCU's rule engine is often directionally correct.

---

## Recipe 6: CSV and raw export

```bash
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page raw --csv \
    > "$PROFILE_RUN_DIR/analysis/raw_<tag>.csv"

ncu --import "$PROFILE_RUN_DIR/reports/source_<tag>.ncu-rep" --page source \
    > "$PROFILE_RUN_DIR/analysis/source_<tag>.txt"
```

The Python API is usually easier, but CSV is useful for quick inspection.

---

## Recipe 7: Targeted metrics only

If you only need a few metrics:

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    gpu__time_duration.sum,\
    l1tex__t_sector_hit_rate.pct \
    -k "regex:.*KERNEL_TOKEN.*" -c 1 \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py" [args]
```

---

## Recipe 8: A/B comparison

```bash
ncu --set full -k "regex:.*my_kernel.*" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/v1" \
    python3 "$PROFILE_RUN_DIR/target/profile_my_kernel_v1.py" [args]

ncu --set full -k "regex:.*my_kernel.*" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/v2" \
    python3 "$PROFILE_RUN_DIR/target/profile_my_kernel_v2.py" [args]
```

Then compare with `analyze_reports.py` or the Python API.

---

## What each `--set` contains

```bash
ncu --list-sets
ncu --list-sections
```

Rough mapping (B200, NCU 2026.1):

| Set | Sections included | Replay passes | Use when |
|---|---|---|---|
| `basic` | SOL, LaunchStats, Occupancy | ~3-5 | Smoke test |
| `detailed` | Middle-ground preset | ~15 | Faster but limited |
| `full` | Everything except Source | ~45 | First-pass profile |
| `source` | SourceCounters included | ~50 | Per-line stall attribution |

---

## Common section additions

```bash
--section PmSampling
--section PmSampling_WarpStates
--section SourceCounters
--section Nvlink_Topology
--section Nvlink_Tables
```

---

## GPU frequency locking

```bash
nvidia-smi -q -d CLOCK
sudo nvidia-smi -lgc <boost_clock_mhz>
sudo nvidia-smi -rgc
```

Usually unnecessary for B200 full profiles, but useful if results jitter across repeated runs.

---

## Gotchas

- **`--set full` is slow**: each replay reruns the target launch.
- **`regex:...` matches nothing**: start from the Python/kernel token, then inspect the captured action name in a report.
- **`regex:...` matches too much**: refine the token to exclude helper kernels or unrelated specializations.
- **Report file is empty or 0 KB**: the target crashed before the selected launch or the regex never matched.
- **PM sampling returns nothing**: check the section flags and the environment.
- **First-run compile polluted the profile**: warm up once outside NCU or skip early launches with `-s`.
- **Source-level report lacks line mapping**: the runtime did not preserve usable source info; keep working at the report and metric level unless the runtime has another way to recover source attribution.
