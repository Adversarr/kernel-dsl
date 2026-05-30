---
name: ncu-report-skill
description: Profile correct CUDA kernels emitted by Python DSLs with Nsight Compute on B200 / sm_100. Use when the user asks to analyze performance, diagnose bottlenecks, read an ncu report, or write an optimization plan for a Python-DSL-generated kernel that already passes correctness checks.
---

# Skill: Python DSL Kernel Profiling (B200 / Nsight Compute)

**When to use:** user asks to profile a CUDA kernel emitted by a Python DSL or runtime-managed stack, analyze its performance, find its bottlenecks, or write an optimization plan based on Nsight Compute data. The kernel must already pass correctness checks before this skill begins. Common triggers include: "profile X", "为什么这个 kernel 慢", "ncu report 说...", "下一步怎么优化", "帮我看一下这份 ncu 报告", or requests involving TileLang, Triton, `torch.compile`, TVM-FFI, or other Python-side compilation flows.

**Target hardware (this repo):** NVIDIA B200 (sm_100, CC 10.0, 148 SMs, 192 GB HBM3e). Most advice below is generic; B200-specific notes are explicitly marked.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing CUDA kernels are under-performing for exactly one reason that ncu can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

For DSL workflows, the target is usually a small Python profile driver that imports the kernel module, compiles or JITs it, warms up if needed, and then launches the steady-state kernel that Nsight Compute should capture.

This skill starts **after correctness**. Do not use it to debug functional bugs, validation mismatches, or reference-check failures.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` at the repo root — **one directory per run**, never reuse an existing one. Each run contains its own `target/`, `reports/`, `analysis/`, and `REPORT.md`. This rule is mandatory in this repo. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** What Python-level kernel symbol or token identifies the generated CUDA kernel? Which dispatch path or specialization? What question do you want answered? If the kernel takes variable-sized inputs, pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Prepare a small Python profile driver** unless the user is already profiling through an existing Python runner. The preferred shape matches `kernel-creator`: importable kernel module plus `profile_<kernel>.py`. The profile driver should compile or JIT the kernel, launch the already-correct kernel in a steady-state way, and expose a stable kernel token for NCU regex capture such as `regex:.*some_kernel.*`. Put it under `profile/<run_name>/target/`. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md) and the template in [`helpers/profile_template.py`](helpers/profile_template.py).

3. **Run two profiles**: `--set full` (with `PmSampling` sections) for the overview, and `--set source --section SourceCounters` for per-line stall attribution. Drive both through the Python profile script, using `-k "regex:.*<kernel_token>.*"` once the token is confirmed. Write outputs to `profile/<run_name>/reports/`. See [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse with `ncu_report`** Python module — not by eye-balling the CLI. Write analysis outputs to `profile/<run_name>/analysis/`. Use the helpers in [`helpers/`](helpers/). See [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Work through the six analysis dimensions.** See [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md). It maps NCU signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `profile/<run_name>/REPORT.md` with evidence-backed recommendations, ranked by expected impact. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | Primary target-preparation guide for small Python profile drivers |
| [`reference/03-collection.md`](reference/03-collection.md) | NCU command recipes for Python-runner collection, regex targeting, PM sampling, and custom sections |
| [`reference/04-python-api.md`](reference/04-python-api.md) | `ncu_report` Python API patterns for reports emitted from DSL-generated kernels |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions: occupancy, balance, stalls, tensor core, timeline, memory |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Pattern → diagnosis → fix. Merges Blackwell programming principles with NCU signals |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report for a Python-runner-driven profile |
| [`reference/08-b200-metric-names.md`](reference/08-b200-metric-names.md) | sm_100 metric names vs older GPUs — many common names are different |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | Permissions, PM sampling gaps, DSL regex pitfalls, and runtime-specific gotchas |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/profile_template.py`](helpers/profile_template.py) | Preferred template for Python DSL profile drivers — import, compile/JIT, warm up, print regex, launch |
| [`helpers/kernel_name_regex.py`](helpers/kernel_name_regex.py) | Tiny helper for turning a kernel token into the canonical `regex:.*token.*` selector |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | Extract key metrics, produce side-by-side comparisons |
| [`helpers/extract_stall_hotspots.py`](helpers/extract_stall_hotspots.py) | Per-line stall aggregation via `action.source_info(pc)` |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII PM-sampling timeline plotter — makes tail effect visible |
| [`helpers/list_flashinfer_workloads.py`](helpers/list_flashinfer_workloads.py) | Optional workload inspector for projects that use flashinfer-trace datasets |
| [`helpers/ncu_utils.py`](helpers/ncu_utils.py) | Shared Python helpers: safe metric access, per-instance extraction, report loading |

---

## Critical lessons (don't skip)

1. **The stock `ncu_profile_skill.md` metric names don't all work on B200.** Names like `smsp__inst_executed_op_global_ld.sum`, `dram__bytes.sum`, `l1tex__average_t_sectors_per_request*.ratio` return `None` on sm_100. Use the sm_100 names in [`reference/08-b200-metric-names.md`](reference/08-b200-metric-names.md) or enumerate via `action.metric_names()`.

2. **Always preserve source mapping if you want source-level analysis.** Without `-lineinfo` or an equivalent generated-source path, ncu's source view is blank and you cannot do per-line stall analysis.

3. **PM sampling is the only way to see tail effects.** Static metrics average over the whole kernel; only the time-series (either `pmsampling:` metrics or the ASCII plotter in `helpers/`) shows the shape of utilization over time.

4. **Load-imbalance on variable-length inputs is often the #1 bottleneck.** If the workload has sequences of varying length, per-SM active-cycle variance will often dwarf every other effect. Always check the input distribution.

5. **NCU's rule engine (`--page details`) already does half the work.** Each rule comes with `Est. Speedup: X%`. Read them first — they often point straight at the answer.

6. **Regex targeting is part of the workflow, not an afterthought.** Most DSLs emit generated CUDA names that still contain the Python/kernel token. Start from that token and use `regex:.*<kernel_token>.*`, then verify the captured name from `action.name()` or a first inspection run before trusting the report.

7. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific metric values. Never write "the profile shows it's memory-bound" — instead, name the two or three metric values that back your conclusion (e.g., "`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` well under 10%, and `long_scoreboard` stalls dominate the pcsamp histogram, so the kernel is **latency-bound on L1**, not DRAM-bandwidth-bound"). Fill in the actual numbers from your report. Specificity is the deliverable.

## Interop with sibling skills

- Follow the file-shape guidance from `kernel-creator`: one importable kernel module, one `test_<kernel>.py`, one `profile_<kernel>.py`, and optional `debug_<kernel>.py`.
- For TileLang-style kernels, treat the Python entrypoint name as the starting capture token and keep profiling in a dedicated runner rather than inside the implementation module.
- This skill remains self-contained. It should describe how to interoperate with those conventions, not import files from sibling skills.

---

## Related skills

- [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) — Blackwell-specific backend optimization guidance for the generated CUDA kernel once NCU has shown what to fix.
