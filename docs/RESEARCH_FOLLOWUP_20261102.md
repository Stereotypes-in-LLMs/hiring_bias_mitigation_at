# Follow-up analysis and merge-precision control

The user authorized this phase on 2026-11-02 after reviewing the proposed research
questions. These are exploratory follow-ups on the development benchmark. The
original selected-checkpoint results and the signs of the newly completed
mechanism-seed sensitivity tables were already inspected. This document is a
prospective execution record for the new analyses, not a preregistration of an
unseen test set.

## Scope

1. Relate attribute-effect tails to decision-threshold crossing using existing
   scores, at the shared SFT-v3 hire count and across the existing 10–35% grid.
2. Compare residual instability overlap and burden at set, CV–job pair and CV
   levels, including the previously identified 35 borderline pairs.
3. Summarize all three completed seeds per post-training method, retaining the
   distinction between real interventions, attribution approximations and
   verified-head subset rankings.
4. Compare unmerged final PEFT adapters with bf16-merged final adapters on the
   same already-merged SFT base. Reuse frozen final checkpoints; do not train.
5. Reproduce a cross-batch baseline discrepancy discovered while checking the
   mechanism summaries, using one short diagnostic on the fourth allocated GPU
   of the seed 42 merge-comparison node.

No new layer-window training, external-data evaluation or performance-kernel
experiment is submitted in this phase. Those decisions follow the evidence from
these controls. Poster content remains unchanged until the new results have
been checked.

## Decision geometry and affected units

Use the canonical op-margin-v2 matching and whole-threshold tie interpolation.
Each model receives its own threshold for a common hire count. Never transfer a
numerical HF threshold into vLLM scores or the reverse. Identify backend and
readout precision in every analysis. Where HF interpretation scores are used,
calibrate their thresholds from that same HF score set.

Report absolute effect quantiles as well as ratios, with p95 alongside p99;
450 pairs leave only a few observations in the extreme tail. Relate effect
size to the two endpoint scores and actual threshold crossing, and repeat the
effect description in rank/quantile space. Distinguish newly unstable sets from
resolved sets instead of reporting only their net difference. These are
descriptive joint analyses and do not establish a causal role for a head.

The existing primary confidence intervals already resample candidates with
their jobs and attribute variants retained. The new aggregation asks a different
question: whether low set-level overlap also implies low CV-level overlap.
For pair/CV membership, compute "any unstable set" separately at the two
neighboring cuts before interpolating. The maximum of already-interpolated set
memberships is not in general that quantity. Retain the canonical descriptive
min/max weighted-Jaccard convention and label it as such.

Leave-one-CV/job-out checks hold original thresholds fixed. They measure the
influence of deleting those observations, not deployment recalibration or a new
sampling interval. Do not call instability systemic exclusion without examining
decision direction and qualification evidence.

## Three-seed mechanism summary

Use the frozen DPO, GRPO-R 4 and decision-distillation checkpoints and the existing
SFT-v3-selected pair sets: 64 pairs in each of three contrasts and 45 in gender.
Verify pair identity and all completion/provenance markers before pooling.
Keep any legacy-cache provenance qualification visible.

Component-profile correlations derive from real mixer/MLP interventions.
L15.H11 and L23.H1 intervention effects and ranks derive from verify.csv and are
ranks among tested heads, whose membership may differ between runs. Attribution
rankings remain separate. Full-grid p50/p99 use all 450 aligned pairs per
contrast and are not estimates from the selected patching subset.

## Merge-precision experiment

Arms: full-layer r32, layers 15–20 r32 and layers 15–20 r163, each at seeds 42/43/44.
Use the declared final 341-step checkpoints. First reproduce the served model's
chain: merge SFT into the bf16 base, then attach the final adapter. Compare that
final adapter unmerged and bf16 merged. Simultaneously activating SFT and the
final adapter on the raw base would be a different experiment.

Fix tokenizer, prompt order, batch composition, attention implementation,
precision and HF scoring backend within each comparison. Record both fp32
decision readout and native bf16 readout if supported, clearly separated. Check
model, adapter and prompt fingerprints. Resumable outputs must reject stale
inputs or changed scoring settings. Do not infer behavioral loss from the
weight-update projection-retention metric.

Pilot: seed 42, all three arms, a deterministic subset of complete CV–job pairs.
Check reconstructed merged weights/scores against the saved served merge, score
coverage, finiteness and measured throughput before the full run. The proposed
pilot is 12 complete pairs, 828 prompts. The initial estimate is 0.3–0.6 used
GPU-h. Full scoring of nine arms in two execution modes is initially estimated
at 12–18 used GPU-h, to be revised from pilot measurements. Record separate
allocated GPU-hours and actual devices used, along with Slurm IDs and explicit
wall limits. Do not automatically retry a failed GPU job before inspecting it.

Analyze both own-rate changes and common-rate instability curves, candidate
paired differences and changes to the full-versus-window comparison. Hold
thresholds fixed within each bootstrap draw and retain canonical tie handling.
Any changed behavioral ordering is an empirical result of the execution paths;
an unchanged ordering does not prove exact numerical equivalence.

The first pilot stopped at a checkpoint-namespace mismatch before scoring.
The runtime text model uses model.* keys while the served multimodal checkpoint
stores the corresponding text parameters under model.language_model.*. The
reviewed correction permits that one-to-one mapping only for the declared Qwen
architecture and retains exact dtype and bitwise weight comparison. The first
attempt is preserved; the successful second attempt has its own output root.

The second pilot passed all three arms: both reconstructed SFT/final weights
match their saved counterparts bit-for-bit, and directly loading the saved final
model reproduces both pilot margin readouts exactly. The pilot measured about
52–59 seconds for each 828-prompt unmerged pass and 44 seconds for each merged
pass. Full jobs are capped at two hours per seed, three arms concurrently on
three of four allocated GH200s. Three seed nodes therefore have a maximum cap
of 24 allocated GPU-hours. Pilot throughput suggests 65–85 minutes per seed;
this remains an estimate until the full pass finishes. Each successful full job
invokes the CPU aggregator with --when-ready; the last completed seed produces
the combined fixed-cut candidate-bootstrap report automatically.

## Targeted batch reproduction

The CPU mechanism audit found that some live patching baselines differ from
their full-grid margin pass despite identical prompt token IDs and adapter
fingerprints. The largest observed endpoint difference is about 15.69 logits in
fresh DPO seed 43, combat explicit. The median discrepancy is much smaller.
Final-prenorm replacement internal controls agree exactly with the B baseline.
Neither fact establishes the cause of the cross-pass discrepancy.

The GPU diagnostic selects four largest-discrepancy and four smallest-discrepancy
pairs in that one checkpoint/contrast. It replays the original full-pass batches,
patching batches and batch-size-one inputs, each twice. This outcome-selected
sample is for reproduction, not a population estimate. Record prompt, adapter,
library/backend and input-layout identities. Use the legacy interpretation PEFT
path; do not substitute the new merged-SFT experiment's model representation.
The diagnostic runs in a separate subprocess/environment on the otherwise unused
fourth GPU of the seed 42 node, with a 15-minute cap and independent completion
status. It does not alter the shared Python environment or the merge workers.

The completed replay ran 90 forward calls. Each original batch context reproduced
its historical margins exactly, and each context repeated exactly in both
orders. The full-pass versus patching batch discrepancy reached 15.6876 logits;
one of the eight outcome-selected diagnostic pairs changed zero-threshold
crossing status. This is evidence of execution-context dependence on the chosen
subset, not an estimate of prevalence or proof of a particular kernel defect.
The diagnostic used FLA 0.6.0 with causal_conv1d unavailable. The separate merge
comparison retains the original environment without the optional FLA path.

## Outputs and publication

Scripts and reports live under scripts/ and reports/research_followup/. Each
report records source hashes and its statistical unit. Do not include raw CV
text in reports or exports. Stage only files belonging to this follow-up; the
repository contains other experiment outputs that are outside this commit.

The final report must distinguish completed CPU findings, passed pilot checks,
queued/running GPU work and completed GPU comparisons. A submitted Slurm job is
not a completed result.
