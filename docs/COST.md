# GPU and time cost

The final analysis compares arms by what they cost as well as by what they fix: the extra
GPU-hours an arm takes over SFT, how much faster one path is than another, how many parameters
a method changes. A number such as "50% faster" or "80% fewer parameters" goes into the paper
only as a ratio of two measured rows, each with its job ID, hardware and workload. This page
says what is measured, how a step gets measured, and how a claim is formed from the tables.

## What is recorded

Every GPU step of a `scripts/cscs/*.sbatch` job runs through `cost <stage> <label> <command>`
(defined in `scripts/cscs/env.sh`, implemented in `src/hiring_bias_mitigation/utils/cost.py`).
It writes one JSON record per step to `$HBM_COST_DIR` (`$XW/cost/records/` on CSCS):

| Field | What it is |
|---|---|
| `stage`, `label`, `arm`, `config`, `command` | `datagen`, `train`, `eval`, `patching` or `other`; what the step did; the arm it belongs to (`HBM_COST_ARM`); its `--config`; the command line |
| `start`, `end`, `wall_s`, `status`, `exit_code` | wall-clock time; `ok`, `failed`, `terminated` (a signal, e.g. the time limit), or `running`, rewritten every minute while the step is alive |
| `gpu.count`, `gpu.used`, `gpu.model`, `gpu.basis` | the GPUs the step's processes actually ran on, from NVML's process list sampled every 10 s (`basis=nvml-processes`), not the GPUs the job held; the NVIDIA driver and CUDA driver versions |
| `gpu_h` | `wall_s` x `gpu.count` / 3600 |
| `gpu.energy_j`, `gpu.util_mean_pct`, `gpu.mem_peak_gib` | NVML energy counter of those GPUs over the step; mean utilisation and peak device memory while the step was on them (for vLLM this includes the memory it reserves up front) |
| `slurm`, `host`, `code`, `software` | job ID, name, partition, node; commit, branch, and whether `src/`, `scripts/` or `configs/` had uncommitted changes; Python, torch, vLLM, transformers, TRL, PEFT and accelerate versions |
| `notes.counters` | added by the code: `requests`, `completions`, `prompt_tokens`, `output_tokens` from every vLLM and transformers generate call |
| `notes.phases` | seconds in `load` (model load, engine start) and `generate` |
| `notes.facts` | an audit's `run_name`, `n_prompts`, `served_model`; for training `params_trainable`, `params_total`, `train_rows`, `train_tokens_per_epoch` (SFT and DPO; not GRPO, whose completions are generated), `effective_batch`, `world_size`, `precision`, `train_steps` of this segment, `global_step`, `train_runtime_s`, `torch_peak_gib`, and `checkpoint_times`, the time each checkpoint was saved |

`scripts/cost_report.py` joins the records with Slurm accounting (`sacct`: what each job held,
whole-node energy) and writes `reports/cost/`. A job that ran before the records existed is
reconstructed from its log: `step` prints `=== <time> <text>` before each step, and each step
is timed from its marker to the next one. A log does not say how many GPUs a step used, so
those steps are charged the job's whole allocation (`gpu_basis=allocated`).

## How to record a step

**A CSCS job.** The shipped sbatch scripts already wrap their GPU steps. Name the arm at
submission, since sbatch passes the environment on:

```bash
HBM_COST_ARM=dpo_v3 CONFIG=configs/mitigation/dpo/<x>.yaml scripts/cscs/submit.sh scripts/cscs/train_dpo.sbatch
```

Without `HBM_COST_ARM`, the arm comes from the rules in `reports/cost/labels.csv`, matched
against the job name, job ID, config, label and command. A new sbatch script wraps each GPU
step the same way, after the fallback line that lets it run where `env.sh` has no `cost`:

```bash
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
cost datagen judge $PY scripts/<x>.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES=$g cost eval audit $PY scripts/run_audit.py --config "$c" &
```

**The v3 runs, which are the ones the poster reports.** Nothing is recorded until this code is
on `main`: most sbatch scripts source main's `env.sh`, and `train_sft.sbatch` and
`train_grpo.sbatch` source the checkout in `$REPO`. So:

1. Merge the recorder into `main` before the first v3 job.
2. A v3 branch merges `main` before it submits anything (or its jobs run from main's
   scripts); otherwise its jobs run, but unrecorded.
3. Give every job of one final model the same `HBM_COST_ARM` (for example `sft_v3`, then
   `dpo_v3` for its data and training), so that `by_arm.csv` adds up that model's cost and
   `checkpoints.csv` gives the GPU-hours to each saved checkpoint of its training.

The v3 data generated before the merge (jobs 5104067 and 5105355) is reconstructed from its
logs; its split by pass, including the biased pass that training does not use, is in
`docs/V3_NEW_DATA.md` on `xinyang/v3-additional-data`.

**Patching and other code outside this pipeline** (part D's activation patching, a probe
sweep, any new intervention) uses the same wrapper, or the in-process form:

```bash
python -m hiring_bias_mitigation.utils.cost run --stage patching --label layer-sweep -- python patch_sweep.py
```

```python
from hiring_bias_mitigation.utils import cost

with cost.track("patching", "layer sweep", arm="patching_sft"):
    ...
    cost.add(forward_passes=n)                      # the work the GPU time bought
    cost.note(params_modified=d_model * n_layers)   # what the intervention changes
```

`track()` is for one process; a program that starts several (torchrun, worker pools) is
recorded whole by the wrapper. For a parameter claim, a patching method notes
`params_modified`: the entries it writes into the model or the activations (a steering vector
added at k layers is k x d_model). Training gets `params_trainable` from the trainer.

**Turning it off.** `HBM_COST=0` runs every command unrecorded. A recorder that fails, or
cannot be imported in an older checkout, prints a line on stderr and runs the command anyway.

## The report

```bash
../venv/bin/python scripts/cost_report.py            # sacct + records + logs -> reports/cost/
../venv/bin/python scripts/cost_report.py --no-sacct # no Slurm here: reuse reports/cost/jobs.csv
```

| File | One row per |
|---|---|
| `steps.csv` | step: source (`record`, `log`, `sacct`), stage, arm, wall time, GPUs and their basis, `gpu_h`, energy, tokens, parameters, whether it counts in its arm's `recipe`, versions, commit |
| `jobs.csv` | Slurm job: allocation (`gpu_h_alloc`, `node_h`), node energy, `gpu_h_steps`, `gpu_h_idle` (jobs with records only) |
| `by_stage.csv` | stage: `gpu_h_measured` and `gpu_h_reconstructed` in separate columns; then `idle` and the total allocation |
| `by_arm.csv` | arm: GPU-hours by stage, `shared_gpu_h`, `recipe_gpu_h`, `excluded_gpu_h` |
| `checkpoints.csv` | saved checkpoint of a recorded training run: GPU-hours to reach it, earlier segments included |
| `throughput.csv` | generation step: requests, tokens, `requests_per_s` over `generate_s`, GPU-seconds per thousand requests, and vLLM's `prefix_cache_hit_rate` (hit / queried prompt tokens) when the engine ran with `disable_log_stats: false` |
| `params.csv` | adapter under `outputs/`: trained parameters against the base model's text parameters |
| `COST.md` | the tables above with the gaps listed |

Everything in `reports/cost/` is regenerated by the script except `labels.csv`, the
hand-written arm, stage and recipe assignment. Edit it when a step lands in the wrong place,
then rerun.

## Reading the numbers

GPU-hours come on three bases, never mixed in one column:

- **Measured** (`gpu_h_measured`): a recorded step, wall time x the GPUs its processes ran on.
  Compare methods on this.
- **Reconstructed** (`gpu_h_reconstructed`): a step rebuilt from a log, wall time x the job's
  allocated GPUs. It is what the step held, and an upper bound on what it used: a one-GPU audit
  on a four-GPU node counts four. Quote it as "allocated GPU-h (reconstructed from Slurm and
  job logs)".
- **Allocated** (`gpu_h_alloc`, the `total allocated` row): what the jobs held. CSCS bills
  whole nodes, four GH200 each, so this is the project's budget. `node_h` is summed over jobs,
  not elapsed time.

**Idle** is allocated minus measured, for jobs with records only: job setup, CPU-only work,
GPUs a one-GPU step left free. It belongs to the project's spending, not to any method's cost.
A reconstructed job's idle time is inside its steps, so the report leaves its idle blank.

Blank means not measured, never zero: GPU energy before records exist, idle for reconstructed
jobs, the recipe of an arm whose data and training ran outside these jobs (the published SFT).
Node energy from sacct is a lower bound when some jobs report none; `COST.md` lists them.

**Arm rows overlap.** A step shared by several arms (`a+b` in `steps.csv`) counts in full in
each, so a row is what reproducing that arm costs and the rows must not be added up. Only
`train_gpu_h` adds up across arms; `shared_gpu_h` is the part of a row that other arms also
count (inside `total_gpu_h`, not on top of it). The project's total is `total allocated`.

**`recipe_gpu_h`** is what producing an arm once takes: its data generation and training,
shared steps in full. Steps that ended normally count, and so do segments cut by the time limit
or a signal, since the next segment resumed from them. Failed, cancelled, killed and
still-running steps do not, nor do checks of the data (direction checks, D4, a redraw
control). A `recipe` cell in `labels.csv` overrides this for a job, such as a cancelled one whose
finished passes a later job reused. `excluded_gpu_h` is what was left out. Development runs get
their own arm (`grpo_dev`).

`gpu_h_not_ok` counts steps that failed, hit the time limit, were cancelled or killed. A
training or generation job resumes from its checkpoints and cached chunks, so this is an upper
bound on waste, not the waste itself.

## Stating a measured claim

- **Speed: the same work done differently.** Two steps that do identical work (the same
  prompts and decoding, or the same training data and schedule) and differ only in a setting,
  the software or the hardware: prefix caching on or off, batch order, one engine or another.
  Faster by `1 - t_new / t_ref`, on `generate_s` for steady-state throughput (it leaves out
  `load_s`, the model load) or on measured GPU-hours end to end. Say which. Run each condition
  at least twice and report the median and range.
- **Methods: cost to reach an effect.** SFT, DPO and GRPO do different work per step (a GRPO
  step generates rollouts; DPO adds a reference pass), so equal steps are not equal work.
  Compare methods from a common starting checkpoint, on the same evaluation, as the GPU-hours
  each needed to reach the checkpoint that was audited (`checkpoints.csv`, data generation
  included), next to that checkpoint's effect. Report the fixed evaluation of a checkpoint
  separately: it is the same work for every arm.
- **Fewer parameters**: `1 - p_new / p_ref`, from `params_trainable` or `params_modified`,
  against the same base (`params_base_text` in `params.csv`).
- Quote both rows' job IDs and the basis (measured or reconstructed).

From the 2026-10-20 report: LoRA r=32 on the seven targeted projections trains 58,195,968
parameters, 0.650% of Qwen3.5-9B's 8,953,803,264 text parameters. That count leaves out the
vision tower (456,010,480) and the multi-token-prediction head (243,290,624), which the
adapters do not train and the audits do not run (`params.csv`). Every 9B English arm uses that
same adapter shape.

## Caveats

- Reconstructed steps charge the whole node, so a one-GPU or CPU-only step inside such a job
  is overstated. Jobs submitted once the recorder is on `main` carry records and are measured.
- `CUDA_VISIBLE_DEVICES` indices are read as NVML indices. On a node of identical GPUs both
  follow the PCI bus order; UUIDs are matched exactly.
- In a container whose PID namespace NVML cannot see into, the step's processes are not
  recognised, and the record counts the visible GPUs instead (`gpu.basis=visible`), kept apart
  from measured use in the report.
- HF's `total_flos` is 0 for TRL's GRPO trainer, so FLOPs cannot compare RL arms with the
  others; GPU-hours and tokens can.
- The published SFT adapter (`Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en`) and the
  v1 data were trained and generated elsewhere (a DGX Spark); their cost is not in these
  tables, and a runtime recovered for them belongs in its own hardware-labelled column.
