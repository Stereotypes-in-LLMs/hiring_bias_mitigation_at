# Acceleration benchmarks: what the NVIDIA stack changes, before and after

Registered 2026-11-02, before any job, at the user's request ("All three … 安排同时跑这三个"):
the GTC reviewers ask how NVIDIA hardware and software enable or speed up the workflow, with a
before-and-after measurement. Branch `xinyang/accel-bench` (from main afd1c25), cost arm
`accel_bench`. Measurement rules are those of [`COST.md`](COST.md), "Stating a measured claim":
steady-state time apart from the model load, every condition at least twice, median and range,
GPU-hours on one basis per column. None of the runs here is audited, scored or kept.

The stack as it stands ([`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md)): one node of 4 × GH200
120 GB per job; vLLM 0.31 for the audits, the student sampling and the GRPO rollouts
(colocated engine, weights synced in fp32); TRL 1.15 / PEFT under `torch.distributed.run`
(DDP) for training, bf16, LoRA r32; the 122B-A10B teacher served in GPTQ-Int4 with tensor
parallel 4; Triton kernels for the GDN prefill. What is already measured: the audits' engine
against cache off and shuffled order, 1.10× on generation ([`THROUGHPUT_B.md`](THROUGHPUT_B.md),
round 2). What is not: any run without vLLM, and any run on fewer GPUs.

## 1. Registration

### B1. The audit's generation: vLLM against transformers' generate

**Question.** How much faster does the audits' engine decide the benchmark than the
repository's own HF transformers loop (`eval/backends.py`, `TransformersBackend`, the path the
embedding mitigation uses), on the same GPU and prompts?

**Arms.** `configs/throughput/accel/qwen3.5-9b_en_accel-{vllm,hf8,hf32}.yaml`: the base model,
the benchmark's first 45 pairs (3,105 prompts, one tenth of an audit), greedy, 512 new tokens,
as-built order. `vllm`: vLLM 0.31, prefix cache on, the audits' engine. `hf8`: transformers
generate, left-padded batches of 8 (the backend's default). `hf32`: batches of 32. One run per
GPU, two repeats, each repeat's conditions on different GPUs (`bench_audit_backend.sbatch`).

**Control.** `hf8` is "before": a plain generate loop. `vllm` is the audits' "after".

**Measured.** `generate_s` (steady-state generation, the model load excluded), `wall_s`, requests
per second, GPU-s per 1,000 decisions, GPU energy, from the `cost` record of each run
(`reports/cost/throughput.csv`, `scripts/throughput_b.py speedup --round accel`). The greedy
decisions of every run against the production base audit on the same rows (`compare --round
accel`): the speed-up must not change the decisions beyond what cache-off runs already showed
(round 2: decision identity in `reports/throughput_b/full/identity.csv`).

**Expected.** vLLM several times faster than `hf8` on `generate_s`; `hf32` between them. Against
the expectation: a ratio below 2×, or decisions that differ on more than 1% of rows.

**Stop.** One job, 3 h cap; a condition that fails twice is reported as failed, not retried.

### B2. Training: one GPU against four under DDP

**Question.** How much does the 4-GPU DDP node buy over one GH200 for a post-training stage, at
the same effective batch?

**Arms.** `configs/mitigation/distill/bench_ddp{4,1}.yaml`: decision distillation seed 42, the
first 40 optimizer steps, 32 rows a step (4 GPUs × accumulation 8, or 1 GPU × accumulation
32), the same data, LoRA and precision as the real run; nothing saved. One after the other on
one node (`bench_train_scaling.sbatch`).

**Control.** The 1-GPU run is "before"; the 4-GPU run is the project's "after".

**Measured.** The trainer's `train_runtime` and `train_samples_per_second` (steady training,
after the model load), and the `cost` record's wall and GPU-h. Speed-up = runtime(1 GPU) /
runtime(4 GPU); scaling efficiency = speed-up / 4.

**Expected.** Speed-up 3–3.8× (DDP with accumulation, LoRA gradients only). Against: below 2.5×.

**Stop.** One job, 1 h 45 cap. A run that stops under `memory_floor_gb` is reported as such.

### B3. GRPO rollouts: the colocated vLLM engine against transformers' generate

**Question.** What does the colocated vLLM engine buy GRPO, whose cost is dominated by sampling
16 completions per prompt?

**Arms.** `configs/mitigation/grpo_improve/bench_rollout_{vllm,hf}.yaml`: R4's recipe
(`r4_tipped.yaml`), six optimizer steps of 256 completions, 4 GPUs under DDP. `vllm`: the
colocated engine, weight sync as in the real run. `hf`: `use_vllm: false`,
`benchmark_generation: transformers`, rollouts through transformers' generate. In both,
`benchmark_only: true` turns off the monitors that generate outside the step (sync validation,
decision monitor), so the step is the same work on both paths. One after the other on one node
(`bench_grpo_rollout.sbatch`).

**Control.** `hf` is "before"; `vllm` is the real run's "after".

**Measured.** `train_runtime` over the six steps (the engine's start-up is before training and
excluded), seconds per step, and the `cost` record's wall and GPU-h, which include the engine's
start-up. Speed-up on both.

**Expected.** 3–10× on `train_runtime`. Against: below 2×. A failure of the `hf` path on this
hybrid model (GDN layers under transformers' generate) is a result too: it says the path is
unavailable, not slow.

**Stop.** One job, 2 h 30 cap; the `hf` run is given at most 90 minutes.

### Order and cost

The three jobs are submitted together (the user, 2026-11-02), one node each. Allocated GPU-h
(4 GPUs × wall): B1 about 8 (the `hf8` runs set the wall), B2 about 4, B3 about 6; about 18 in
all, under the `accel_bench` arm. The three sbatch scripts take `HBM_COST_ARM=accel_bench
REPO=$PWD`.

### B2 repeat: the 1-GPU run once more, with the GPU's clocks logged (registered 2026-11-02)

**Why.** The first 1-GPU run slowed from 50 to 62 s a step after step 5 while the 4-GPU run
held 13.7 s; the 4.40× mixes the two phases (section 3). The user: rerun the baseline rather
than word around it, and log throttle reasons.

**Run.** `scripts/cscs/bench_train_1gpu_repeat.sbatch`: the same `bench_ddp1.yaml` on GPU 0,
node allocated whole (`--exclusive`; the first run's node was too: 288 CPUs, all memory), cost
label `bench-distill-ddp1-r2`. Every 10 s: SM and memory clocks, power, temperature,
utilisation (`logs/xw-bench-1gpu-<job>-gpu.csv`) and `nvidia-smi -q -d PERFORMANCE`, whose
"Clocks Event Reasons" name any throttle (`-perf.txt`); the PERFORMANCE, CLOCK and POWER
sections before and after.

**Read as.** No slowdown (steps 6–40 within 5% of steps 1–5): the first run's drift was a
one-off; the poster's ratio becomes runtime(repeat) / runtime(4 GPUs), stated with the drift
of the first run in this document. The slowdown again: the poster reports both phases and
the throttle reason the log names, or "no throttle reason logged" if none.

**Stop.** One job, 1 h 15 cap, about 1 GPU-h of the 4 allocated used.

## 2. Ledger

| # | job | what | GPU-h allocated | running total |
|---|---|---|---|---|
| 1 | 5236801 | B3, both paths (nid006001), 13:55 wall: vLLM rollouts `train_runtime` 183.5 s, cost step 557 s (0.619 GPU-h measured); transformers rollouts `train_runtime` 200.5 s, cost step 268 s (0.298 GPU-h). 1,536 completions each (6 steps × 256), mean length 37 tokens, none clipped (`logs/xw-bench-grpo-5236801.out`) | 0.93 | 0.93 |
| 2 | 5236800 | B2, both runs (nid006496), 55:50 wall: 4 GPUs under DDP `train_runtime` 549.6 s, 2.33 samples/s, cost step 850 s (0.945 GPU-h measured, 562 kJ); 1 GPU `train_runtime` 2,420 s, 0.53 samples/s, cost step 2,489 s (0.691 GPU-h, 453 kJ); 40 steps of 32 rows each, train_loss 0.066 / 0.070 (`logs/xw-bench-train-5236800.out`) | 3.72 | 4.65 |
| 3 | 5236799 | B1, six runs (nid006001 node group, one GPU each), 43:03 wall: `vllm` `generate_s` 64.4 / 65.0 s (load 74 / 86 s, wall 150 / 161 s, 0.0126 kWh); `hf8` 1,065 / 1,117 s (load 15 s, wall 1,085 / 1,137 s, 0.0739 kWh); `hf32` 460 / 480 s (load 16 s, wall 481 / 502 s, 0.0488 kWh); 3,105 decisions each, all decided (`logs/xw-bench-audit-5236799-*.log`, `reports/throughput_b/accel/`) | 2.87 | 7.52 |
| 4 | 5238558 | B2 repeat, 1 GPU (nid006338, node exclusive), 38:02 wall: `train_runtime` 2,040 s, 0.627 samples/s, cost step 2,271 s (0.631 GPU-h measured, 421 kJ); steps 1–5 57.4 s/step (step 1 64 s), steps 6–40 50.1 s/step; GPU 0 at 1,980 MHz SM throughout, 185–195 W of a 900 W limit, 26–27 °C, utilisation 17–21% by quarter; throttle counters unchanged over the job (`logs/xw-bench-1gpu-5238558*`) | 2.54 | 10.06 |

## 3. Result

### B1. The audit's generation: vLLM against transformers' generate

**Fact.** On the same 3,105 prompts and one GH200 each, steady-state generation took 64.7 s
under the audits' engine (vLLM 0.31, prefix cache on; two repeats, 64.4 and 65.0 s), 1,091 s
under transformers' generate in batches of 8 (1,065 and 1,117 s) and 470 s in batches of 32
(460 and 480 s): **16.9×** against `hf8`, 7.3× against `hf32`; `hf32` alone is 2.3× over
`hf8`. Requests per second 48.0 against 2.85 and 6.61; GPU-seconds per 1,000 decisions
(generation only) 20.8 against 351 and 151; GPU energy per run 0.0126 against 0.0739 and
0.0488 kWh (5.9× less than `hf8`). The engine's start-up is longer, 80 s against 15 s of model
load, so end to end the ratio on this one-tenth audit is 7.1× (wall 156 against 1,111 s); on a
full audit the start-up is paid once for ten times the prompts. Prefix cache hit rate 17.6%
(408 k of 2.31 M prompt tokens).

**Decisions.** The two vLLM repeats decide identically to each other and to the production base
audit on all 3,105 rows (text identical on 92.7%, the cache and order effects of round 2). The
transformers runs differ from vLLM on 7–10 rows (99.68–99.78% the same decision), from each
other on 1 row (`hf8` r1 against r2) and across batch sizes on 10–11 rows; the hire/reject
direction of the differing rows is balanced (4 against 3). Transformers' raw output ends in a
newline that vLLM strips, so `same_text` between the two backends is 0 by construction; the
JSON inside is the same (`reports/throughput_b/accel/identity.csv`).

**Against the registration.** Several times faster, as expected, and the decisions differ on
0.2–0.3% of rows, under the 1% registered as counting against.

**Interpretation, not confirmed.** Continuous batching with paged KV cache keeps the GPU busy
while static left-padded batches wait for their longest sequence; `hf32` recovering 2.3× by
batch size alone says much of `hf8`'s loss is occupancy. The prefix cache's share is known from
round 2 (1.10×), not separated here.

**Unknown.** Two repeats each on different GPUs of one node; transformers at batch 64 or with
sorted lengths; the ratio for the student models and for the 122B teacher.

### B2. Training: one GPU against four under DDP

**Fact.** The same 40 optimizer steps of 32 rows took 2,420 s on one GH200 (accumulation 32)
and 549.6 s on four under DDP (accumulation 8): **4.40×** on `train_runtime`, 0.53 against 2.33
samples per second. In GPU-hours the four-GPU run cost 0.945 against 0.691 (its wall includes
four processes' model loads), and in GPU energy 562 against 453 kJ: 37% more GPU-h for 4.4×
less wall time. Train loss over the 40 steps 0.066 and 0.070. One run each.

**Against the registration.** At or above the expected 3–3.8×: the ratio exceeds the GPU
count; scaling efficiency on `train_runtime` 110%.

**Per step** (`scripts/accel_step_times.py`, from the progress bar in the log;
`reports/throughput_b/accel/step_times.csv`). The two configurations are the same except the
accumulation count: one row per GPU and micro-step, gradient checkpointing on in both, global
batch 32 rows. The 4-GPU run took 14.0 s a step over steps 1–5 and 13.7 s over steps 6–40. The
1-GPU run took 49.6 s a step over steps 1–5 and then 62.0 s over steps 6–40. Over the same
phases the ratio is **3.5×** (steps 1–5) and **4.5×** (steps 6–40); the 4.40× on
`train_runtime` mixes the two.

**Cost beside speed** (the user, 2026-11-02: report the two together). Used GPU-h of the two cost
steps, GPUs × wall with process start and model load included: 0.945 (4 GPUs, 850 s) against
0.691 (1 GPU, 2,489 s), **+37%** for the 4.4× in wall-clock. On training time alone the order
reverses for this pair of runs: 4 × 549.6 = 2,198 GPU-s against 2,420 (−9%), because of the
1-GPU run's slowdown. The difference is outside training: 300 s of each 4-GPU process's step
(850 − 549.6) against 69 s (2,489 − 2,420), 1,202 against 69 GPU-s. What those 300 s are (four
concurrent model loads, process-group start-up, data preparation per rank) is not separated.
The poster states the wall-clock ratio with the used-GPU-h change of the same two steps.

**Interpretation, not confirmed.** The above-linear ratio comes from the 1-GPU run slowing by
a quarter after step 5, not from any difference in work per row; at its early pace the scaling is
sublinear (3.5×, 88%). What slowed it (clocks, power, memory, the host) was not measured; the
end of warm-up (step 5) is the only change in the schedule there. An earlier reading here, that
the 1-GPU run pays more per row, is withdrawn: its work per micro-step is the same.

**Unknown.** One repeat each; the ratio at other per-device batch sizes.

### B2 repeat: the 1-GPU run once more, with the GPU's clocks logged

**Fact.** The repeat ran the same 40 steps in 2,040 s (`train_runtime`; the first run 2,420 s):
57.4 s a step over steps 1–5, of which step 1 took 64 s, then 50.1 s over steps 6–40, 50.5 over
6–20 and 49.7 over 21–40. No slowdown. On GPU 0, sampled every 10 s over the 35 minutes of
training: SM clock 1,980 MHz in every sample (its maximum), power 185–195 W by quarter against
a 900 W limit, 26–27 °C, utilisation 17–21%. Every "Clocks Event Reasons" counter (SW power
cap, HW and SW thermal slowdown, power brake, sync boost) was unchanged between the snapshots
before and after the job, and the only active reason in 224 snapshots was Idle, at the ends.
Peak device memory 23.1 GiB (the first run: 30.7; the 4-GPU run: 23.6 per GPU). NVML mean
utilisation of the whole step: 18.4% (the first 1-GPU run 16.4%, the 4-GPU run 25.1%).

**Against the registration, and how the rule was applied.** The rule was written to catch a
slowdown: "No slowdown (steps 6–40 within 5% of steps 1–5)". Read two-sided it fails: steps
6–40 are 12.7% *faster* than steps 1–5, because step 1 carries the warm-up. Read as the
slowdown test it was written as (steps 6–40 at most 5% slower), the repeat passes. The
generator applies the one-sided reading (`poster/latex/latex_data.py`, `TrainState`), so the
repeat is the 1-GPU baseline. This reading was settled after seeing the repeat; the two-sided
reading would have kept the first run and reported both phases.

**Result with the repeat as baseline.** One basis per column:

| Basis | 1 GPU (repeat) | 4 GPUs | Speed-up | GPU-h, 4 against 1 |
|---|---:|---:|---:|---:|
| Training (`train_runtime`; GPUs × it) | 2,040 s | 549.6 s | **3.71×** | +8% (2,198 against 2,040 GPU-s) |
| Whole cost step (start and load included) | 2,271 s | 850 s | 2.67× | +50% (0.945 against 0.631 GPU-h) |
| Steady steps (6–40, per step) | 50.1 s | 13.7 s | 3.66× | |

Scaling efficiency on training time 93%.

**Interpretation, not confirmed.** The GPU is idle four fifths of the time in every B2 run:
one row per micro-step with gradient checkpointing is a sequence of small kernels, and the
log warns that the GDN layers' `causal_conv1d` falls back to its PyTorch reference
implementation. A step bound by the host and by kernel launches would scale with the number of
processes as much as with the number of GPUs, which fits near-linear scaling at 18–25%
utilisation. Which of the two dominates is not measured. The first run's slowdown after step 5
remains unexplained: it had no clock log, its peak memory was 7.6 GiB higher, and it ran on GPU
0 right after the 4-GPU run in the same job; none of these is tested.

**Unknown.** One repeat; the effect of installing `causal_conv1d` or of larger per-device
batches (more rows per micro-step) on both the ratio and the utilisation.

### B3. GRPO rollouts: the colocated vLLM engine against transformers' generate

**Fact.** Over six steps of 256 completions, the vLLM path trained in 183.5 s and the
transformers path in 200.5 s (`train_runtime`, the engine's start-up excluded): **1.09×**, 30.6
against 33.4 s per step. End to end, the vLLM run's cost step took 557 s and the transformers
run's 268 s: the colocated engine's start-up and first weight sync cost about 370 s, so for six
steps the path without vLLM was faster. Both runs sampled completions of 36–38 tokens on
average, none truncated at 96, and rewards in the same range (0.65–1.04 per step). One run per
path.

**Against the registration.** The expected 3–10× did not appear; the observed ratio is below
the 2× that was registered as counting against. At R4's real length, 259 steps, the per-step
saving (2.8 s) would repay the start-up after about 130 steps and end about 350 s (4%) ahead.

**Per step** (`scripts/accel_step_times.py`, the trainer's `step_time`): 29.7 s on the vLLM path
and 32.5 s on the transformers path, completions of 37 tokens on average on both. The colocated
engine saves 2.8 s a step.

**Interpretation, not confirmed.** The rollouts here are prefill-dominated: prompts of about
2,500 tokens, completions of under 40. Transformers' generate on a GH200, 64 sequences per
process, is close to vLLM on that shape; vLLM's advantage lies in long decodes. The weight sync
every step and the engine's 0.30 memory share may also cost the vLLM path. Which of these holds
is not measured.

**Unknown.** One repeat per path; whether `generation_batch_size` or `vllm_gpu_memory_utilization`
change the ratio; the ratio at longer completions.

## 4. Reading

## 5. Caveats

## 6. Reproducing

```bash
HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh --exclude=nid005679,nid006377 scripts/cscs/bench_audit_backend.sbatch
HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh --exclude=nid005679,nid006377 scripts/cscs/bench_train_scaling.sbatch
HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh --exclude=nid005679,nid006377 scripts/cscs/bench_grpo_rollout.sbatch
# afterwards, on the login node
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 ../venv/bin/python scripts/cost_report.py
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 ../venv/bin/python scripts/throughput_b.py speedup --round accel
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 ../venv/bin/python scripts/throughput_b.py compare --round accel
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 ../venv/bin/python scripts/accel_step_times.py
```
