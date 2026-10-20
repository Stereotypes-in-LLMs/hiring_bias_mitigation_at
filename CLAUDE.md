# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An audit-and-mitigate study of hiring bias in open-weight LLMs (Qwen3.5-4B/9B, gemma-4-E4B/12B,
LAPA-12B) in English and Ukrainian. A counterfactual benchmark of 450 job-CV pairs is decided
many times, identical except for one injected protected attribute (military status, gender,
religion), under explicit and implicit injection plus an attribute-free control. Five
mitigation families (prompt, scrub, embedding erasure, SFT, DPO/KTO) are measured against that
baseline on matched rows. The headline metric is counterfactual set stability.

`README.md` is the full manual. `docs/METRICS.md` defines every number and what it misses.
`reports/PAPER_KIT.md` maps each claim to the file that backs it. Read those before changing
how anything is measured.

## Commands

Run everything through `uv run`. Calling `.venv/bin/python` directly leaves the venv's `bin/`
off `PATH`, and Triton's JIT then fails with `FileNotFoundError: 'ninja'`. Scripts and tests put
`src/` on `sys.path` themselves.

```bash
make local-setup                                   # uv venv 3.12 + deps except torch; install torch for the machine after
make local-test                                    # pytest tests/ -v; fast, no GPU, no network, real vendored data
uv run pytest tests/test_eval.py -k "set_stability" -v   # one test
make lint                                          # ruff check src/ scripts/ tests/
uv run python scripts/preflight_training.py        # builds every enabled trainer's args without a GPU; run before any training stage
./scripts/quickstart.sh --tier smoke --local       # 20 pairs, one model, minutes; run before any multi-hour sweep
```

The pipeline, in order. `make <target>` runs in Docker, `make local-<target>` runs in the venv
and sources `.env`.

```bash
make local-benchmark      # validate the vendored benchmark, write data/benchmark/holdout_ids.json
make local-audit          # stage 1: baseline audit of every enabled config
make local-report         # regenerate reports/RESULTS.md from eval/results/*.json
make local-analyze        # grade the evidence -> reports/ANALYSIS.md, mitigation_plan.yaml, enable_planned.sh
make local-plan           # enumerate planned runs -> reports/EXPERIMENT_PLAN.md (reads reports/analysis.json)
make local-generate       # teacher-model training data -> $HBM_OUTPUT_ROOT/artifacts/semisynthetic-v1
make local-prompt         # then local-scrub, local-embedding, local-sft, local-dpo, in that order
make progress             # which config is running, how far, how many runs are scored
make watch                # tail the newest log with -n 0 (vLLM's progress bar is one multi-megabyte line)
```

One run by hand:

```bash
uv run python scripts/run_audit.py --config configs/audit/qwen3.5-9b_en_baseline.yaml
uv run python scripts/run_audit.py --config <cfg> --score-only --similarity-device cpu   # re-score cached generations; no GPU, seconds
uv run python -m hiring_bias_mitigation.mitigation.sft --config configs/mitigation/sft/<x>.yaml
uv run python -m hiring_bias_mitigation.mitigation.dpo --config configs/mitigation/dpo/<x>.yaml
uv run python scripts/fit_eraser.py --config configs/mitigation/embedding/<x>.yaml      # before an embedding audit
```

Derived tables and figures. These only read `eval/results/` and the raw parquets, so they are
safe to run while a sweep is in flight.

```bash
uv run python scripts/set_stability.py                      # reports/set_stability*.csv, mitigation_decision_table.csv
uv run python scripts/reference_metrics.py --all            # one row per run + verdict -> reports/reference_metrics.csv, comparisons.csv; also written by every audit
uv run python scripts/merge_adapter.py --config <audit cfg> --retention-only   # share of the update a merge kept, for merges older than the measurement
uv run python scripts/operating_point_control.py margins --config <audit cfg> --which base|adapter
uv run python scripts/operating_point_control.py compare --config <audit cfg>            # stability at equal hire rate
uv run python scripts/diagnose_adapter.py --help            # HF+PEFT decision logit against the vLLM audit, row by row
uv run python scripts/cost_report.py                        # GPU and time cost by step, stage and arm -> reports/cost/ (docs/COST.md)
uv pip install -e '.[viz]' && uv run python scripts/make_figures.py --only tradeoff --formats png
```

Selecting what runs:

```bash
uv run python scripts/generate_experiment_configs.py --enable smoke|stage1|stage1-all|core|full|none
uv run python scripts/generate_experiment_configs.py --no-runners     # rewrite YAMLs only; required while a sweep is running
uv run python scripts/select_configs.py --runner scripts/run_all_sft.sh <config...>    # exact selection in one runner
./scripts/queue_after.sh <config...>                                  # append to an in-flight sweep without touching its runner
./reports/enable_planned.sh                                           # apply the selection analyze_results produced
```

`HBM_KEEP_GOING=1` (or `--keep-going`) makes an unattended sweep log failures and continue
instead of stopping at the first one.

## Architecture

### The pipeline has an order, and stage 1 is read by a human

```
benchmark -> audit (stage 1) -> report + analyze -> [decide what to mitigate]
          -> generate -> prompt -> scrub -> embedding -> sft -> dpo -> audit the adapters
```

Which cells get a mitigation run is decided by reading `reports/ANALYSIS.md`, not by a preset.
Zero-training families run first because they set the bar a training run has to beat. DPO
continues from the matching SFT checkpoint, so SFT-only against SFT+DPO is an ablation on
identical data.

### Configs are generated, cross-referenced, and relocated

`scripts/generate_experiment_configs.py` writes every YAML under `configs/` and the
`CONFIGS=(...)` array in each `scripts/run_all_*.sh`. That array is the single source of truth
for what runs, however the sweep is launched. Edits to a generated file are overwritten; change
`MODELS`, `AUDIT_BASE` or a writer function instead. A hand-written arm survives regeneration
only if the generator does not emit a file of that name. The header of
`configs/mitigation/sft/qwen3.5-9b_en_only_v2.yaml` shows the convention for such a file.

`utils/config.py::load_config` splices referenced configs in. A mitigation config's
`audit_config` becomes `cfg["audit"]` and a training config's `data_config` becomes
`cfg["data"]`, so a runner receives one dict. Configs hold short relative paths such as
`outputs/sft/x`, and `resolve_output_path` puts them under `$HBM_OUTPUT_ROOT`. Training and
generation refuse to start unless `$HBM_OUTPUT_ROOT/.hbm-models-root` exists; set
`HBM_ALLOW_LOCAL_OUTPUT=1` to bypass on a machine with no model drive.

`configs/data/*.yaml` are views over one generated dataset (language, group, condition,
`limit_rows`), so a per-group or per-language arm trains without regenerating data.
`reports/eval_scope.yaml` pins which groups and conditions each model-language is evaluated on.
The generator applies it and `tests/test_mitigation.py` checks that every family honours it.

### A run is one cell, and everything it produces hangs off its name

`eval/runner.py::build_run_name` gives `<Model>--<lang>--<family|baseline>--<variant>[--<run_suffix>]`,
for example `Qwen3.5-9B--en--sft--adapter--en_only`. That name keys:

| What | Where |
|---|---|
| every response unmodified, plus the parse columns (`decision`, `outcome`, `raw_output`) | `$HBM_OUTPUT_ROOT/outputs/raw/<run>.parquet` and `<run>.meta.json` (model, `served_model`, decoding, seed, `config_path`, wall time) |
| the scored record: per-attribute rows with denominators, raw and FDR-adjusted p, bootstrap CIs, a manual-review queue | `eval/results/<run>.json`; `index.json` is the inventory |
| rendered tables | `reports/RESULTS.md` |
| a trained checkpoint and what produced it | `$HBM_OUTPUT_ROOT/outputs/{sft,dpo}/<name>/run_config.json` |

Generations are written before any scoring, which is what makes `--score-only` cheap. The
only difference between a baseline run and a mitigated run is the `mitigation` block. The
docstring of `eval/runner.py` says where each family intervenes. Embedding erasure is a forward
hook and needs `backend: transformers`. Adapters are merged into the weights before vLLM serves
them (`mitigation/merge.py`), because vLLM's LoRA path does not reproduce Qwen3.5 adapters.
The audits that took that path are kept in `eval/results_vllm_lora_invalid/` as a record, not
as results.

### Derived files are regenerated, never hand-edited

`reports/RESULTS.md`, `ANALYSIS.md`, `EXPERIMENT_PLAN.md`, `mitigation_plan.yaml`,
`enable_planned.sh`, `experiment_runs.csv`, `set_stability*.csv`,
`mitigation_decision_table.csv`, `training_curves.csv`, everything under `figures/`,
`eval/results/index.json`, `reports/reference_metrics.csv`, `reports/reference_metrics/`,
`reports/comparisons.csv`, everything under `reports/cost/` except the hand-written
`labels.csv`, and every generated config. The phase-1 versions of the
analysis and plan outputs are kept unchanged in `reports/archive/phase1/`. `FINDINGS.md`,
`MITIGATION_FINDINGS.md`, `PAPER_KIT.md`, the model cards and `docs/*.md` are written by hand
from those files, and each number in them names the file it came from.

### Comparisons are matched

A baseline covers the full grid, 179 attributes including the intersections. A mitigation run
covers its scoped cells. Every comparison restricts both sides to the same rows, cells and
attribute variants first (`analysis/stability.py`, `eval/report.py`). Set stability is paired
per set, tested with an exact sign test under Benjamini-Hochberg, with a bootstrap over
candidates. A fairness change is read beside reference agreement and the attribute-free hire
rate, and `scripts/operating_point_control.py` compares two models at equal hire rate because
instability depends on the operating point.

### Contamination control

`data/benchmark/holdout_ids.json` (300 candidates, 301 jobs) is subtracted in
`generation/pool.py`, re-checked by `assert_no_leakage`, respected by the candidate-wise
train/validation split, and checked again by the publish scripts. Do not bypass a leakage
error. Never train on anything under `outputs/raw/`; those are benchmark outputs.

### Tests pin past mistakes

Many tests exist because of a specific wrong result. Matched-scope deltas, adapter audits
refusing the vLLM LoRA path, Ukrainian targets using the prompt's own decision words, trainer
arguments constructing for every enabled config. When a methodological bug is fixed, add the
test that would have caught it.

## Evidence-first experiment workflow

Several clean, significant, plausible results in this repository turned out to be artefacts.
The five are written up under "Methodological lessons" in `reports/MITIGATION_FINDINGS.md`.
The rules below are how a round of work is specified, reported and followed up.

How two methods are compared is fixed in `docs/EVAL_STANDARD.md` and coded in
`analysis/eval_standard.py`: the matched-hire-rate decision, its guardrails, the protocol, and
the per-run reference metrics. Evaluate a new arm with `scripts/cscs/evaluate_arm.sbatch`, read
its verdict from `reports/reference_metrics.csv`, and change the standard only on the user's
decision.

### Before running anything

Write down, in the task description or the write-up draft:

1. The question this round answers, and the main hypothesis.
2. The control. Name the exact baseline run and the matched scope the comparison is on.
3. The expected observation if the hypothesis holds, and what would count against it.
4. The stop condition. Examples in use here are the parse-failure abort threshold in
   `scripts/run_audit.py`, `early_stopping_patience`, `memory_floor_gb`, a data gate on the
   share of `chosen` that says hire, and a GPU-hour cap.
5. Which split selects the model and which is the final test. The generated pool's
   `*_validation.parquet` (split by candidate) and `eval_loss` select checkpoints and
   hyperparameters. The 450-pair benchmark audit is the final test and is spent once per
   configuration. A checkpoint chosen with audit numbers in view is tuned on the test set;
   if that happened, the card says so.

### After running, deliver the observation report first

Before any interpretation:

- A one-sentence conclusion, with observed fact, interpretation and unknowns in separate
  sentences.
- The key change against the baseline on matched rows. Change in unstable sets with its CI
  and fixed/broken counts, change in reference agreement, change in attribute-free hire
  rate. Take them from `reports/set_stability.csv` or `mitigation_decision_table.csv`.
- The figure or table, and the path to per-sample results (the raw parquet, or the CSV
  beside the figure).
- Raw cases. A few actual responses from `outputs/raw/<run>.parquet` that show the
  behaviour, quoted as written.
- Where evidence is missing, which evidence, and why.

Every number links to a run name, an `eval/results/*.json`, a CSV under `reports/`, or a
job or log ID. No number is typed from memory or computed outside the repository's scripts.
A performance gain is not evidence that the proposed mechanism caused it.

Detailed logs go to a file (`logs/`, the W&B run, the scheduler's job output) and are
referenced by path. The main reply does not narrate the execution step by step.

### After the user's preliminary judgment, deliver the recommendation report

Only then:

- List what this round's evidence supports and what it does not.
- When more than one explanation fits, state each and propose the cheapest check that tells
  them apart, with its cost. Cheap checks here include a `--score-only` re-score, a
  `diagnose_adapter.py` pass, a cross-seed rerun of a short config, and an operating-point
  comparison.
- Include at least one alternative that challenges the current line, rather than assuming
  the next step is more training. A data fix, a different control, or stopping the arm are
  all valid proposals.
- Without confirmation, do not change evaluation definitions (`eval/metrics.py`,
  `analysis/stability.py`, the thresholds in `scripts/analyze_results.py`, decoding
  settings), the holdout or the train/validation split, or `reports/eval_scope.yaml`. Do not
  enable more configs, epochs, seeds or grid than agreed.

The separation, on a small example:

> Fact. With the learning rate halved, the validation metric on the held-out split improved.
> Interpretation. The original updates may have been too large. Not confirmed.
> Discriminating checks. Compare update magnitudes between the two runs; repeat on two more
> seeds; list which benchmark decisions changed and in which direction.

### The result card

Each experiment gets a write-up at `docs/<EXPERIMENT>.md` with the sections Setup, Result,
Reading, Caveats and Reproducing. The card at the top links at least these:

| Field | Where it lives here |
|---|---|
| code version and uncommitted diff | `code.commit` and `code.uncommitted` in the op-margin-v2 JSON under `reports/operating_point/v2/`, and the `commit` field on the first line of a `scripts/cscs/*.sbatch` job log. The audit `meta.json` and the training `run_config.json` do not record it, so for those write `git rev-parse --short HEAD` and `git status --porcelain` into the card by hand |
| full config and data version | the config path (`meta.config_path`, `run_config.json`), `data_config`, `artifacts_dir` such as `artifacts/semisynthetic-v1`, and `data_kind` |
| training run ID | `output_dir` under `$HBM_OUTPUT_ROOT/outputs/{sft,dpo}/`, the checkpoint step, `wandb_run_name`, the scheduler job ID |
| independent evaluation run ID | the audit run name and its `eval/results/<run>.json`; `served_model` in the meta shows the merged checkpoint that was actually evaluated |
| figures and per-sample results | `figures/{en,uk}/<figure>.{svg,csv}`, `reports/set_stability*.csv`, `outputs/raw/<run>.parquet` |
| GPU and time cost | the run's job IDs in `reports/cost/steps.csv` (`gpu_h`, `wall_s`, `params_trainable`), recorded by `cost <stage> <label>` in the sbatch scripts; `docs/COST.md` |

### Traps this project has already fallen into

- Comparing unmatched populations. Lessons 1 to 3 were each a clean, significant effect
  that vanished on matched rows, cells and variants.
- Trusting the serving stack. Seven adapters showed a consistent null through vLLM's LoRA
  path, and the evaluated model was not the trained one. Before reading a null for a trained
  model, check HF+PEFT agreement with the audit on a sample of prompts.
- Training on a normalised value. Ukrainian targets carried `reject` where the prompt asks
  for `відхилити`. Parse failures stayed at 0% and fairness improved. Read a sample of
  training targets as text before training on them.
- Reading a flag count as an effect size, or a per-run roll-up as a claim about a group.
  Argue from the model x language x group x condition row.
- Selecting a checkpoint on the audit. Both the best-loss and the final checkpoints are on
  disk. Audit the one chosen in advance, and label any choice made with audit numbers in view.
