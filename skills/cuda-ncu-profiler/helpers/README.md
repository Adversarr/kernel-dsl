# Helpers

Reusable code for Python DSL profile drivers and Nsight Compute report analysis. See `../SKILL.md` for context.

## Python / DSL

| File | Purpose |
|---|---|
| `profile_template.py` | Preferred starting point for a Python profile driver. Import the kernel, compile or JIT it, print the regex token, then launch. |
| `kernel_name_regex.py` | Tiny helper for building the canonical `regex:.*token.*` selector. |
| `ncu_utils.py` | Shared helpers: `load_report`, `safe`, `per_pc_values`, `B200_KEY_METRICS`, and report-loading utilities. |
| `analyze_reports.py` | Extract key metrics and side-by-side comparisons from one or more `.ncu-rep` files. |
| `extract_stall_hotspots.py` | Aggregate per-PC stall samples into per-source-line rankings. |
| `plot_timeline.py` | ASCII plot PM sampling timelines. |
| `list_flashinfer_workloads.py` | Optional workload inspector for projects that use flashinfer-trace datasets. |

### Typical DSL-first workflow

```bash
export PROFILE_RUN_DIR=profile/<run_name>
HELPERS=/path/to/skills/cuda-ncu-profiler/helpers
export FIB_DATASET_PATH=/path/to/flashinfer-trace

mkdir -p "$PROFILE_RUN_DIR"/{target,reports,analysis}
cp "$HELPERS/profile_template.py" "$PROFILE_RUN_DIR/target/profile_<kernel>.py"

# Optional: inspect the workload space
python3 "$HELPERS/list_flashinfer_workloads.py" --definition <def_name>

# Collect
ncu --set full --section PmSampling --section PmSampling_WarpStates \
    -k "regex:.*<kernel_token>.*" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    python3 "$PROFILE_RUN_DIR/target/profile_<kernel>.py"

# Extract key metrics
python3 "$HELPERS/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag>
```

`ncu_utils.py` tries to auto-locate `ncu_report` from common CUDA install paths. If that fails, set `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python
```
