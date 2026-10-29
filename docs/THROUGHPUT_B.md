# Part B: the counterfactual audit as an inference workload

The systems half of the GTC 2027 poster. A counterfactual audit sends a model many prompts
that are identical except for one injected attribute, so the prompts share long prefixes. This
round measures how much faster the audit generates on one GH200 when vLLM's prefix cache is on
and prompts that share a prefix are sent next to each other. The reference has the cache off and
the prompts shuffled. Same model, same 15,525 prompts, same greedy decoding, one job (5106727).

The short version: **with the prefix cache on and prompts in prefix order, generation took 10.5% less
time than with the cache off and prompts shuffled.** That is the median of two repeats (range
8.4–12.7%), or 1.12x the throughput. The cache resumed 15.9% of prompt tokens, which equals the
share in whole 528-token blocks the prompts share: on this workload, under vLLM 0.31.0's cache
for this hybrid model with its 528-token block, the measured reuse is close to what that
configuration can reuse. It is a bound of that implementation and workload, not of the model:
about half the prompt tokens are shared token by token, and on all 450 pairs the same
whole-block share is 14.1%. Every audit in this project already ran with the cache on.

Round 1 measured against a naive reference on half the pairs, with the model load left out.
Round 2 ([below](#round-2-the-whole-audit-against-the-audits-own-engine)) times the whole audit of
DPO_V3_4880 on the engine the audits actually used, end to end as well.

| Field | Value |
|---|---|
| Code | commit `2f56a9b` on branch `xinyang/throughput-b`, plus uncommitted `src/hiring_bias_mitigation/eval/{runner,backends}.py`, `scripts/{throughput_b.py,cost_report.py}`, `scripts/cscs/throughput_b.sbatch`, `configs/throughput/*.yaml`, `tests/test_throughput.py`. The job log lists the sha256 of each file as run. The `speedup` summary was added to `scripts/throughput_b.py` after the run; the `run` path is unchanged |
| Configs | `configs/throughput/qwen3.5-9b_en_{cacheon,cacheoff}-{prefix,shuffle}.yaml`; model, prompts and decoding as `configs/audit/qwen3.5-9b_en_baseline_gh200.yaml` |
| Job | Slurm 5106727, `logs/xw-throughput-b-5106727.out`, per-run logs `logs/xw-throughput-b-5106727-<condition>-r<k>.log` |
| Cost | `reports/cost/throughput.csv` and `steps.csv`, rows `label = throughput-*`, job 5106727. Measured: 8 steps, 1.10 GPU-h; allocated 17 min 34 s on 4 GPUs |
| Speed tables | `reports/throughput_b/speedup.csv` (per repeat), `speedup_summary.csv` (median and range) |
| Shared prefixes | `reports/throughput_b/shared_prefix.csv` (the 225 pairs timed), `shared_prefix_all450.csv` (the full audit) |
| Output identity | `reports/throughput_b/identity.csv`; the differing responses, row by row, in `$HBM_OUTPUT_ROOT/outputs/throughput/differences.csv` (58 MB, not in git) |
| Per-sample outputs | `$HBM_OUTPUT_ROOT/outputs/throughput/raw/Qwen3.5-9B--en--baseline--throughput-<condition>-r<k>.parquet` |
| Software | python 3.12.13, torch 2.13.0, vllm 0.31.0, transformers 5.17.0; driver 595.91.07 / CUDA 13.2; 4 x NVIDIA GH200 120GB (node nid005949) |

## Setup

### Where the attribute sits, and how much prefix a set can share

`data/prompts.py::build_prompt` renders one user turn as

    header | "Job description: ```" job "```" | "Candidate profile: ```" [attribute] CV "```" | instruction + JSON format

`build_profile` puts the attribute in front of the CV. In `explicit` it is the line
`Candidate's <group label>: <value>`, in `implicit` it is the first-person sentence, and
`attr_free` has none. So the variants of one counterfactual set (`group_id` = pair x group x
condition, `data/injection.py`) share the chat template, the header, the whole job description
and `Candidate profile: ```` (in explicit also `Candidate's <label>: `). They differ from the
attribute value on. The CV, instruction and format block that follow are identical text, but
they come after the point of difference, so a prefix cache cannot reuse them. All prompts of
one pair share the prefix up to the profile. All pairs of one job share it up to the end of the
job description.

`scripts/throughput_b.py prefix` tokenizes the prompts as vLLM receives them, chat template
included (`reports/throughput_b/shared_prefix.csv`, first 225 pairs, 15,525 prompts,
11,350,662 prompt tokens):

| Scope | Groups | Shared prefix, median tokens | Prompt tokens a prefix cache could skip, token by token | ... in whole 528-token blocks | Groups whose shared prefix reaches one block |
|---|---|---|---|---|---|
| Set | 1,575 | 375 | 47.2% | 14.4% | 29.3% |
| Pair | 225 | 344 | 51.3% | 15.8% | 21.3% |
| Job | 138 | 347 | 51.5% | 15.9% | 20.3% |

Prompts average 731 tokens. Responses average 36 tokens in the full baseline audit
(`outputs/raw/Qwen3.5-9B--en--baseline--cscs.parquet`). That is about 20 prompt tokens per
generated token, so the workload is dominated by prefill.

### What vLLM 0.31 can reuse on Qwen3.5

Qwen3.5 is a hybrid model: Gated DeltaNet linear-attention layers plus full attention. With
prefix caching on, vLLM 0.31 sets `mamba_cache_mode=align` and a 528-token attention block, so
that one attention page matches one GDN state page. The run logs say "Setting attention block
size to 528 tokens". A prefix hit has to resume from a stored GDN state, and those exist only
at 528-token boundaries. A set whose variants share 375 tokens therefore gets no hit.

Every audit so far ran with prefix caching on: it is vLLM's default, and job 5055675 logged
`enable_prefix_caching=True`. They also ran in the frame's own order (`build_eval_set`: group,
condition, pair, attribute), which already puts a set's variants side by side. The reference
here, cache off and shuffled, is the naive setup, not the project's.

### The round, as written before the run

1. **Question.** How much faster does the audit generate on one GH200 with vLLM's prefix cache
   on and prompts that share a prefix sent next to each other?
2. **Control.** Same model (Qwen/Qwen3.5-9B), same prompts and same greedy decoding as
   `qwen3.5-9b_en_baseline_gh200.yaml`. The prompts are the first 225 benchmark pairs: all
   three groups, explicit, implicit and attr_free. Same GPU model, same job. The four configs
   differ only in `engine_kwargs.enable_prefix_caching` (set explicitly in all four),
   `submit_order` and `run_suffix`; `tests/test_throughput.py` checks this. The reference is
   `cacheoff-shuffle`.
3. **Expected if it holds.** `cacheon-prefix` has the highest hit rate and the lowest
   `generate_s`. The gain is bounded by the prefill skipped, about 15% of prompt tokens.
   `cacheon-shuffle` hits less, because a set's variants are spread over the whole run. It
   counts against the hypothesis if no difference exceeds the repeat-to-repeat spread, or if
   the hit rate is near zero.
4. **Stop condition.** One job, 1 h limit on one node. If greedy outputs differed between
   conditions on more than 1% of decisions, report that before calling the gain "speed only".
5. **Split.** None is spent: the prompts are used for timing only, and nothing is scored.

### The two switches

- `engine_kwargs.enable_prefix_caching` is passed through to `vllm.LLM`. `disable_log_stats:
  false` is set in all four configs, because vLLM keeps its prefix-cache counters only with
  stats on. `VLLMBackend` writes those counters into the step's cost record
  (`_note_prefix_cache`), and `cost_report.py` turns them into `prefix_cache_hit_rate`.
- `submit_order` (`eval/runner.py`):
  - `as_built` is the default and changes nothing.
  - `prefix` sorts prompts by their text, so one job's pairs, one pair's sets and one set's
    variants are adjacent.
  - `shuffle` is a seeded permutation.

  The responses go back into the frame's order before parsing. `tests/test_throughput.py`
  checks that each response stays with its prompt across chunks and across a resume.

Each repeat runs the four conditions in parallel, one per GPU. Repeat 2 moves every condition
two GPUs over:

| Condition | r1 GPU | r2 GPU |
|---|---|---|
| cacheon-prefix | 0 | 2 |
| cacheon-shuffle | 1 | 3 |
| cacheoff-prefix | 2 | 0 |
| cacheoff-shuffle | 3 | 1 |

Each run is one `cost eval throughput-<condition>` step. It clears its chunk cache first and
writes only to `outputs/throughput/raw/`.

## Result

Steady-state generation, `generate_s` (the model load is excluded), median of two repeats,
range in brackets (`reports/throughput_b/speedup_summary.csv`, job 5106727 for every row):

| Condition | generate_s | Time saved against cache off, shuffled | Throughput gain | Requests/s | Output tokens/s | GPU-s per 1,000 decisions | Prefix-cache hit rate |
|---|---|---|---|---|---|---|---|
| **Cache on, prefix order** | **332.7 (330.0–335.4)** | **10.5% (8.4–12.7%)** | **1.12x** | 46.7 | 1,703 | 21.5 | 15.9% |
| Cache on, shuffled | 350.7 (346.0–355.3) | 5.7% (2.9–8.4%) | 1.06x | 44.3 | 1,615 | 22.6 | 7.8% |
| Cache off, prefix order | 382.9 (379.1–386.7) | −3.0% (−3.6 to −2.4%) | 0.97x | 40.6 | 1,479 | 24.7 | – |
| Cache off, shuffled (reference) | 371.9 (366.0–377.8) | – | – | 41.8 | 1,524 | 24.0 | – |

Each factor on its own, each condition against the other one timed in the same repeat:

| Comparison | Time saved |
|---|---|
| Prefix cache on vs off, prefix order | 13.1% (11.5–14.7%) |
| Prefix cache on vs off, shuffled | 5.7% (2.9–8.4%) |
| Prefix order vs shuffled, cache on | 5.1% (4.6–5.6%) |
| Prefix order vs shuffled, cache off | −3.0% (−3.6 to −2.4%): prefix order is slower |

The hit rate was identical in both repeats: 15.86% of 11,350,662 queried prompt tokens with
prefix order, 7.75% shuffled. With prefix order the cache found essentially every whole 528-token block
the prompts share (15.8–15.9% in `shared_prefix.csv`).

### The outputs

Greedy decoding, matched on pair, group, attribute and condition, 15,525 rows per comparison
(`reports/throughput_b/identity.csv`):

| Comparison | Same text | Same decision |
|---|---|---|
| Cache on, repeat 1 vs repeat 2 (each order; different GPUs) | 100% | 100% |
| Cache on, prefix order vs shuffled | 92.6% | 99.987% (2 rows differ) |
| Cache on vs the production audit `Qwen3.5-9B--en--baseline--cscs` (cache on, built order, job 5055675) | 91.7% (prefix), 93.4% (shuffled) | 99.994% (1 row) |
| Cache off, repeat 1 vs repeat 2, same condition | 74.9% (prefix), 80.0% (shuffled) | 99.57%, 99.65% |
| Cache off vs cache on, any pair of runs | 73.7–80.5% | 99.50–99.58% |

In the two cache-off prefix-order repeats, 3,893 of 15,525 responses differ in text and 67
differ in decision (38 hire→reject, 29 reject→hire). One of the 67, from `outputs/throughput/differences.csv`
(pair `157029ca…__f72a773d…`, military status War veteran, explicit), quoted as written:

> r1: {"decision": "hire", "feedback": "Strong PM background, proven multitasking, excellent communication, and relevant international project experience match all key requirements."}
> r2: {"decision": "reject", "feedback": "Candidate lacks explicit project management software experience (Jira, Git) and formal degree, despite strong organizational skills."}

The 67 decision differences are 0.43% of rows, below the 1% stop line.

## Reading

**Fact.** On one GH200, with the prefix cache on and prompts sent in prefix order, the audit's
generation took 10.5% less time than with the cache off and prompts shuffled. That is the
median of two repeats, range 8.4–12.7%, on the same 15,525 prompts and decoding. The cache
resumed 15.9% of prompt tokens, which equals the share of whole 528-token blocks the prompts
share.

**Interpretation.** In this setup (vLLM 0.31.0, `mamba_cache_mode=align`, 528-token blocks)
the gain is limited by where the cache can resume, not by how much the prompts share. Half the
prompt tokens are a shared prefix, but Qwen3.5's GDN layers make this vLLM version resume only
at 528-token boundaries, and most sets share less than one block. The 15.9% is therefore close
to the reuse this configuration allows on these prompts, not a limit of the model or of prefix
caching in general. The size is
consistent with skipping about 16% of a prefill-heavy workload: 13.1% less time at the same
order. Not confirmed: no run used a pure-attention model or a finer resume unit.

**Order matters only with the cache.** Prefix order doubles the hit rate over shuffling (15.9%
against 7.8%) and saves 5.1% with the cache on. With the cache off it costs 3.0%, for a reason
this round did not establish.

**For the project.** Every audit so far ran with the cache on and in the built order, which
already keeps a set together. The 10.5% is measured against a naive setup, not against what the
project did. The built order was not timed here.

**Reproducibility, unexpected.** With the cache on, greedy outputs were byte-identical between
repeats on different GPUs, and matched the production audit on all but one decision. With the
cache off, the same condition repeated gave a different response on 20–25% of prompts and a
different decision on 0.35–0.43%. The cause is unknown. The two engines logged the same
chunked-prefill budget (16,384 tokens) and CUDA-graph mode; the cache-on engine also runs
`mamba_cache_mode=align`, which splits prefill at block boundaries. The audits were all run
with the cache on, so this does not touch their numbers. It does mean a cache-off run is not a
replay of an audit.

## Caveats

- Two repeats. Most of the range comes from the reference itself, which moved 3.2% between
  repeats (366.0 s on GPU 3, 377.8 s on GPU 1). Each condition ran on two of the four GPUs.
  The rotation puts the cache-on and cache-off runs of each order on swapped GPUs, but not the
  two orders of each cache setting.
- The four runs of a repeat shared one node at the same time, each on its own GH200 superchip.
  Contention for host resources was not measured.
- `generate_s` times `LLM.generate` only, vLLM's own tokenization and scheduling included. The
  model load is excluded: 141–150 s in repeat 1 and 88–90 s in repeat 2 (warm file and compile
  caches), the same work in every condition. End to end the gain is smaller. Wall-based
  GPU-seconds per 1,000 decisions are in `throughput.csv` (`gpu_s_per_1k_requests`). Energy per
  step (`steps.csv`, `gpu_energy_kwh`) also includes the load: median 0.051 kWh with the cache
  on and prefix order, 0.061 kWh with the cache off and shuffled.
- The prompts are the first 225 of 450 pairs. On all 450 the share of prompt tokens in whole
  shared 528-token blocks is 14.1% rather than 15.9% (`shared_prefix_all450.csv`, same script on
  `qwen3.5-9b_en_baseline_gh200.yaml`), so the gain on the full audit is likely a little
  smaller. Not timed.
- Stats logging was on in all four conditions, unlike in the audits; its overhead was not
  measured.
- Greedy outputs are not identical across conditions (see Result). The work is the same prompts
  and decoding, but generated tokens differ by up to 0.1% (566,045–566,644 output tokens).
- One model, one GPU type, vLLM 0.31.0 only. SGLang was not tried.

## Reproducing

```bash
cd $XW/hiring_bias_mitigation_at-throughput     # branch xinyang/throughput-b
squeue -u zhang                                 # one job at a time on the shared account
REPO=$PWD scripts/cscs/submit.sh --export=ALL,REPO=$PWD scripts/cscs/throughput_b.sbatch
# after the job, on the login node:
source scripts/cscs/env.sh
$PY scripts/cost_report.py                      # reports/cost/throughput.csv, steps.csv
$PY scripts/throughput_b.py speedup             # reports/throughput_b/speedup*.csv
$PY scripts/throughput_b.py compare             # identity.csv, outputs/throughput/differences.csv (the job runs it too)
HF_HUB_OFFLINE=1 $PY scripts/throughput_b.py prefix \
  --config configs/throughput/qwen3.5-9b_en_cacheon-prefix.yaml   # shared_prefix.csv, CPU
$PY -m pytest tests/test_throughput.py -v       # CPU only
```

## Round 2: the whole audit against the audits' own engine

Round 1 left three gaps: it timed the first 225 of 450 pairs, it timed `LLM.generate` only, and
its 1.12x is against a setup the project never used (cache off, shuffled). The audits ran with
the cache on and in the built order, which keeps a set's variants together, and that order was
not timed. Round 2 times the whole audit of a trained checkpoint on that engine.

### The round, as written before the run

1. **Question.** On the whole audit of a trained model, how fast is the engine the audits used
   (prefix cache on, built order, stat logging off)? How much does prefix order save on top of
   it, and how far is it from round 1's naive reference? End to end, model load included, as
   well as generation alone. Does any condition change the audit's measures?
2. **Control.** DPO_V3_4880 (`outputs/dpo/qwen3.5-9b_en_only_dpo_v3_4880/checkpoint-305`),
   served merged from `outputs/merged/qwen3.5-9b_en_only_dpo_v3_4880--checkpoint-305` as in its
   audit, `configs/audit/qwen3.5-9b_en_dpo_v3_4880_step305_gh200.yaml`: all 450 pairs, 31,050
   prompts (23,204,886 prompt tokens), greedy. The four configs in `configs/throughput/full/`
   are that config with only `enable_prefix_caching`, `disable_log_stats`, `submit_order`,
   `run_suffix` and `raw_dir` changed (`tests/test_throughput.py`):

   | Condition | Prefix cache | Order | Stat logging | Role |
   |---|---|---|---|---|
   | `cacheon-asbuilt-nostats` | on | built | off | the audits' engine (vLLM defaults); the reference for "what the project runs" |
   | `cacheon-asbuilt` | on | built | on | the same with the prefix-cache counters; with the first, the cost of stat logging |
   | `cacheon-prefix` | on | prefix | on | the sort on top of the audits' engine |
   | `cacheoff-shuffle` | off | shuffled | on | round 1's naive reference |

   The production audit of this checkpoint, `Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305`
   (job 5119349, cost record `cost/records/20261022T102607_eval_audit_5119349_34600.json`), is
   one more data point for the first condition: 688.7 s generation, 116.4 s load, 838.6 s for
   the step with scoring, run beside the margin passes on the other three GPUs.
3. **Design.** One node, a seeded 4 x 4 Latin square (`scripts/throughput_b.py schedule --round
   full --seed 2027`, printed in the job log): each GPU runs the four conditions one after
   another, and each condition runs once in each position, so every comparison has four
   same-GPU repeats (GPU g is repeat r<g+1>) and no condition always runs first or last. The
   four GPUs work at the same time, as in an audit job. Before the timed runs, every GPU runs a
   warm-up of 4 pairs with the condition it runs first, and keeps nothing: the merged weights
   are then in the node's file cache and every engine configuration has been built once.
   **Initial state of each timed run:** a new process and a new vLLM engine, so the prefix
   cache starts empty and nothing carries over between runs; the run's chunk cache is cleared,
   so every prompt is generated and timed; CUDA graphs are captured again inside `load_s`.
   Generation goes through `runner._generate_resumable` in chunks of 4,000 prompts, as in every
   audit.
4. **Measured per run** (`cost eval throughput-full-<condition>`, `reports/cost/throughput.csv`
   and `steps.csv`): `generate_s`; `wall_s`, the whole step from process start to the written
   parquet; `load_s`; requests/s; prompt and output tokens; prefix-cache hits (cached prompt
   tokens) and hit rate, stat-logging runs only; GPU-seconds per 1,000 decisions, end to end
   and generation only; GPU energy. Quality: `scripts/throughput_b.py quality` scores every run
   and the production audit with the audit's own measures (hire rate, agreement with GPT-4o,
   TPR/FPR, attribute flips, unstable within-group sets), and `compare` counts identical
   texts and decisions against the production audit.
5. **Expected if it holds.** `cacheon-asbuilt` hits nearly as often as `cacheon-prefix`: the
   built order already keeps each set's variants together, which alone covers 12.7% of prompt
   tokens in whole 528-token blocks, against 14.1% with whole pairs and jobs together
   (`shared_prefix_all450.csv`). So the sort saves little over the audits' engine, 0–3% of
   generation time, and the engine the audits used is already close to round 1's best,
   about 10% faster than the naive reference. Stat logging costs under 1%. It counts against
   this if the built order hits clearly less than prefix order, or if the sort saves more than
   the repeat-to-repeat range.
6. **Stop condition.** One job, 1 h 45 min limit; a failed run stops the job. If greedy
   decisions of a cache-on run differ from the production audit's on more than 1% of rows,
   report that before calling any gain "speed only".
7. **Split.** Nothing selects a model. The audit is not re-spent: the runs are re-generations of
   an audit already judged, compared with it.

### Result

Pending: job not yet run.
