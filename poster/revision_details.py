"""Rebuild the LaTeX poster's QR evidence page from completed result tables."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/"
METHODS = [("DPO", "DPO_V3_4880", "dpo4880"),
           ("GRPO + lr decay", "GRPO+decay R4", "grpo_r4"),
           ("Self-distillation", "Distill", "distill43")]


def read(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def link(name):
    return f"[`{name}`]({REPO}{name})"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                     "| " + " | ".join(["---"] * len(headers)) + " |"] +
                    ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def write(path=ROOT / "poster/DETAILS.md"):
    summary = json.loads((ROOT / "reports/poster_followup/summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("CPU evidence is incomplete")
    compute = json.loads((ROOT / "reports/poster_followup/compute_summary.json").read_text())
    window = json.loads((ROOT / "reports/poster_followup/window_budget_summary.json").read_text())
    if not window["complete"] or window["seeds"] != [42, 43, 44]:
        raise ValueError("Window parameter-budget evaluation is incomplete")
    for name, expected in window["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale window summary; rerun poster/latex/window_data.py: {name}")
    window_primary, window_common = [], []
    guardrails = json.loads((ROOT / "reports/poster_followup/window_budget_guardrails.json").read_text())
    if not guardrails["registered_trainer_settings_identical"] or len(guardrails["records"]) != 9:
        raise ValueError("Window training/quality checks are incomplete")
    window_checks = []
    for row in guardrails["records"]:
        if row["completed_global_step"] != 341 or row["last_logged_training_loss"]["optimizer_step"] != 340:
            raise ValueError("Unexpected checkpoint or training-loss scope")
        window_checks.append([row["seed"], row["arm"],
                              f"{row['last_logged_training_loss']['value']:.5f}",
                              f"{row['merge_kept_pct']:.3f}%",
                              f"{row['quality']['agree_gpt4o_pct']:.3f}%",
                              f"{row['quality']['agreement_delta_vs_same_seed_full_pp']:+.3f}"])
    for row in window["rows"]:
        for arm, label in (("window_r32", "Window r32, 11.4M"), ("window_r163", "Window r163, 58.1M")):
            p, c = row["primary"][arm], row["common"][arm]
            window_primary.append([row["seed"], label, f"{p['hire_rate_pct']:.3f}%",
                                   f"{p['delta_window_minus_full_pp']:+.3f}",
                                   f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]"])
            window_common.append([row["seed"], label, f"{c['delta_arm_minus_full_pp']:+.3f}",
                                  f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"])
    m = compute["macros"]
    final = read("reports/final/final_table.csv")
    curves = read("reports/final/curves.csv")
    ref = {float(r["hire_rate_pct"]): float(r["unstable_pct"]) for r in curves if r["label"] == "SFT-v3"}
    primary, curve_rows, tails = [], [], []
    for name, prefix, spec in METHODS:
        rows = [r for r in final if r["label"] in [f"{prefix} s{s}" for s in (42, 43, 44)]]
        if len(rows) != 3:
            raise ValueError(f"Missing primary seeds: {name}")
        for r in sorted(rows, key=lambda r: int(r["seed"])):
            primary.append([name, r["seed"], f"{float(r['delta_common_pp']):+.3f}",
                            f"[{float(r['ci_common_low']):+.3f}, {float(r['ci_common_high']):+.3f}]"])
        per_seed = [{float(r["hire_rate_pct"]): float(r["unstable_pct"]) for r in curves
                     if r["label"] == f"{prefix} s{s}"} for s in (42, 43, 44)]
        if any(set(seed) != set(ref) for seed in per_seed):
            raise ValueError(f"Curve grid mismatch: {name}")
        below = [sum(seed[x] < ref[x] for x in ref) for seed in per_seed]
        mean_below = sum(sum(seed[x] for seed in per_seed) / 3 < ref[x] for x in ref)
        curve_rows.append([name, ", ".join(f"{n}/{len(ref)}" for n in below),
                           f"{mean_below}/{len(ref)}"])
        for r in read(f"reports/interp/distill/{spec}/sensitivity.csv"):
            tails.append([name, r["contrast"], r["pairs"],
                          f"{float(r['r_attr_median']):.3f}", f"{float(r['r_attr_p99']):.3f}"])
    overlaps = read("reports/seed_overlap/overlap.csv")
    own = [[name, r["seeds"], f"{float(r['jaccard']):.3f}", f"{float(r['sft_v3_jaccard']):.3f}"]
           for name, prefix, _ in METHODS for r in overlaps
           if r["recipe"] == prefix and r["seeds"] in ("42/43", "42/44", "43/44")]
    common = [[r["recipe"], r["seeds"], f"{float(r['weighted_jaccard']):.3f}", "1.000"]
              for r in read("reports/poster_followup/common_rate_overlap.csv")]
    effect = [[r["contrast"], r["pairs"], f"{float(r['removed_pct']):.1f}%",
               f"[{float(r['ci_low']):.1f}, {float(r['ci_high']):.1f}]%"]
              for r in read("reports/poster_followup/full_grid_effects.csv")]
    heads = [[r["spec"], r["contrast"], r["target"], r["real_rank_tested"], r["verified_heads"],
              f"{float(r['real_removed_logits_per_pair']):.3f}"]
             for r in read("reports/poster_followup/target_heads.csv")
             if r["spec"] in ("dpo4880", "grpo_r4", "distill43") and r["verified"] == "True"]
    text = f"""# Evidence for the revised research poster

This page supports **What Changes After SFT? Auditing Hiring Decisions and Internal Pathways**.
Rebuild it with `python poster/revision_details.py` after regenerating the CPU and compute summaries.
The tables below use completed measurements, including all three registered window-r163
parameter-budget controls. The mechanism panels retain their stated selected-checkpoint scope;
additional mechanism-seed results have not been incorporated into those panels.

## Scope and interpretation

The development benchmark contains 150 candidates, each paired with three distinct jobs,
giving 450 CV–job pairs, 2,700 attribute sets and 31,050 prompts. There are 171 distinct jobs
overall, rather than three jobs shared by every CV. Each set varies identity cues while fixing qualifications, job, attribute family
and cue style. An unstable set includes both hire and reject decisions. These outcomes audit
consistency; GPT-4o agreement is a reference proxy, not independently labelled hiring quality.
Training-candidate exclusion does not make a repeatedly inspected development benchmark a clean
held-out test. The results are exploratory, English-only, on one 9B model and one shared SFT-v3
checkpoint. Post-SFT recipes have three training seeds. No held-out result is claimed.

The principal comparison fixes the hire rate at SFT-v3's 22.512%. Per-seed paired 95% intervals
resample candidates. The CPU breakdown reuses the canonical matching and tie rules, with global
cuts fixed before subgroup analysis and percentile candidate bootstrap intervals conditional on
the evaluated seeds. No three-seed aggregate confidence interval is claimed. Source hashes and tie handling
are recorded in {link('reports/poster_followup/summary.json')} and
{link('scripts/poster_audit_followup.py')}.

## Training signals

- DPO uses 4,880 preference pairs constructed from SFT-v3's sampled mistakes and teacher-written
  chosen responses. See {link('docs/V3_NEW_DATA.md')}.
- The plotted GRPO R4 runs use 4,154 tipped prompts, fixed attribute-free decision-reference
  rewards plus a format reward, and KL regularization to SFT-v3. Learning-rate decay covers
  the final 26% of the schedule. The 9,140-row released GRPO pool is not the training-row count
  of these runs. The optional operating-point-anchor method is not enabled in this recipe.
  See {link('configs/mitigation/grpo_improve/r4_tipped.yaml')} and
  {link('src/hiring_bias_mitigation/mitigation/grpo.py')}.
- Self-distillation uses frozen SFT-v3 as the teacher on the same CV–job pair without identity
  cues. It minimizes teacher-to-student KL over the two decision-token probabilities;
  attribute-free student rows are also included. See
  {link('src/hiring_bias_mitigation/mitigation/decision_distill.py')}.

## Behavioral results and operating points

At 22.51% hire rate, Base-to-SFT instability falls from 11.8% to 1.8%, a change of
−9.95 pp with paired 95% CI [−12.07, −7.83]. GPT-4o agreement rises from 78.8% to 82.4%
at each model's own greedy rate. These two comparisons use different rate conventions.

{table(['Post-SFT recipe', 'Seed', 'Change vs SFT-v3, pp', 'Paired 95% CI'], primary)}

One DPO seed alone would have read as about a 1 pp gain: seed 42's own-rate comparison is
−1.03 pp, while its fixed-22.51% comparison is −0.815 pp. The poster's operating-point curves
show all 51 grid points from 10% to 35%, all three seeds, and their pointwise mean.
Fractions below SFT are fractions of these grid points, not hire-rate ranges.

{table(['Recipe', 'Below SFT, seeds 42 / 43 / 44', 'Seed mean below SFT'], curve_rows)}

The curve points are correlated descriptive evaluations. Their count is not a significance test.
The thin lines are individual seed trajectories, not confidence bands. Paired bootstrap
intervals are displayed only at the fixed 22.51% operating point in the adjacent forest plot.
Sources: {link('reports/final/final_table.csv')}, {link('reports/final/curves.csv')}.
The family × cue-style breakdown is in {link('reports/poster_followup/family_cue.csv')}.
It uses global matched-rate cuts before stratification, not a new threshold for each subgroup.

## Residual instability and tail sensitivity

The poster illustrates the 43/44 seed pair at each arm's own rate. All pairs are shown here.
The control holds SFT-v3 scores fixed and recalibrates its cut to each seed's greedy hire
count, including the canonical weighted tie rule. It does not reuse the post-trained model's
numeric logit threshold. See {link('scripts/seed_overlap.py')} and
{link('scripts/scale_only_control.py')}. A low actual overlap alone cannot separate a recipe
change from a threshold change.

{table(['Recipe', 'Seeds', 'Actual Jaccard', 'SFT threshold control'], own)}

Source: {link('reports/seed_overlap/overlap.csv')}. A second analysis fixes every arm at
the same 22.512% hire rate. Fractional tied-set membership uses sum(min weights)/sum(max weights).
The SFT control is then identical across seeds and equals one. Do not mix actual overlaps from
this table with the own-rate controls above.

For the illustrated seeds 43/44, changing SFT's thresholds alone leaves substantially more
overlap than the trained-seed comparison. This does not rule out every calibration or
operating-point explanation and does not establish that training creates new real-world bias.

{table(['Recipe', 'Seeds', 'Common-rate weighted Jaccard', 'Same-cut SFT control'], common)}

Source: {link('reports/poster_followup/common_rate_overlap.csv')}.
The 35 pairs near SFT's cut are 7.8% of the 450 pairs. The share of remaining instability on
these pairs depends on rate convention: the fixed-rate values are recorded per seed in
{link('reports/poster_followup/common_rate_seeds.csv')}.

The tail plot uses one selected checkpoint per method and all 450 pairs for each contrast.
Its p50 and p99 are ratios of quantiles of absolute attribute-induced margin differences,
relative to SFT-v3. The connecting line spans four contrasts; it is not a confidence interval.
At these selected checkpoints, DPO and GRPO have p50 ratios below one and p99 ratios above one
in all four contrasts. Median absolute attribute effects decrease while the extreme tail
increases. A uniform positive rescaling of all absolute effects cannot produce both changes.
These logit effects are not probabilities of discrimination or errors.

{table(['Recipe', 'Contrast', 'Pairs', 'p50 ratio', 'p99 ratio'], tails)}

Sources: the `sensitivity.csv` files under `reports/interp/distill/` and
{link('reports/poster_followup/tail_sensitivity.csv')}. This behavioral tail result is not
established as a consequence of the patched head changes.

## What the interventions establish

The Base-to-SFT panels select pairs by Base effect. The post-SFT comparisons select on SFT-v3
effect, with 64 pairs per contrast except gender, where 45 pairs pass selection. "Same pairs"
applies within a comparison, not across those panels. The Base-selected combat-implicit
intervention at L15.H11 removes about 0.66 logits per pair in Base and 0.13 after SFT.

Selected-pair effects do not estimate the full benchmark effect. For the implicit male/female
contrast, the Base-effect-selected subset shows about 72% reduction, while all 450 pairs show
about 25% reduction, with a paired candidate-bootstrap 95% CI of 12%–37%. The direction remains
a reduction on the full benchmark, but the estimated magnitude is sensitive to effect-based
selection. The poster displays this full-grid result prominently beside the selected subset.
The full-grid results are:

{table(['Contrast', 'Pairs', 'Mean absolute effect removed', 'Candidate-bootstrap 95% CI'], effect)}

Source: {link('reports/poster_followup/full_grid_effects.csv')}.
Component-profile correlations come from real mixer/MLP interventions in `real.csv`.
Head rankings first use attribution patching in `atp.csv`; `verify.csv` intervenes on a selected
subset of heads. The ranks below are among that tested subset, not all heads in the model.

{table(['Model', 'Contrast', 'Head', 'Real rank', 'Heads tested', 'Logits removed / pair'], heads)}

DPO's combat-implicit L23.H1 is first among 46 tested heads. This supports that specific
checkpoint/contrast finding, not a global later route or a complete causal circuit.
Some random-control intervals are wide, especially gender, so controls cannot all be described
as negligible. Sources: {link('reports/poster_followup/head_verification.csv')},
{link('reports/poster_followup/target_heads.csv')}.

## Completed parameter-budget control for window LoRA

The original layers 15–20 window at rank 32 has 11,403,264 trainable parameters;
the full-layer r32 adapter has 58,195,968. The registered window r163 has
58,085,376 saved parameters, 0.1900% below full, within the declared 0.2% tolerance.
All three training and dependent evaluation jobs completed successfully. Each uses the declared
final 341-step, one-epoch checkpoint, the same data and training settings as its full-layer
seed, with alpha/r fixed at two. Layer placement and rank differ, so matching parameter count
does not equalize optimization geometry, compute, training time or representational capacity.
This is self-distillation with LoRA restricted to layers 15–20, compared with full-layer LoRA.

The primary comparison is window minus same-seed full-layer instability, at the full arm's
own greedy hire count. Positive is worse. Each 95% interval uses 1,000 paired candidate
bootstrap draws with thresholds fixed. The old r32 window is recomputed on this same basis:

{table(['Seed', 'Window arm', 'Full-arm hire rate', 'Change vs full, pp', 'Paired 95% CI'], window_primary)}

The descriptive three-seed mean gap falls from
{window['seed_means_pp']['primary']['window_r32']:+.3f} pp to
{window['seed_means_pp']['primary']['window_r163']:+.3f} pp. The matched-budget estimates
are positive for all three seeds, but only {window['matched_seed_cis_above_zero']['primary']}/3
intervals exclude zero. There is no aggregate three-seed CI and no equivalence test.
The poster's earlier +0.53 pp used a different operating-point convention; it must not be
compared directly with the new +0.39 pp. The displayed +0.60 → +0.39 pp uses one shared convention.

At the secondary common SFT-v3 hire count of 6,990/31,050, or 22.512%, the mean gaps are
{window['seed_means_pp']['common']['window_r32']:+.3f} pp and
{window['seed_means_pp']['common']['window_r163']:+.3f} pp. All three matched-budget intervals cross zero:

{table(['Seed', 'Window arm', 'Change vs full at 22.512%, pp', 'Paired 95% CI'], window_common)}

The larger window adapter reduces the observed average gap. These results neither establish
equivalence with full-layer adaptation nor a general failure of localization-guided editing.
This is a registered follow-up on the exploratory development benchmark, not held-out confirmation.
The experiment is motivated by the distinction between localization and editing in Hase et al. [4].

Training, merge and reference-agreement checks:

{table(['Seed', 'Arm', 'Last logged loss, step 340', 'Merge projection retained', 'GPT-4o agreement', 'Agreement vs full, pp'], window_checks)}

Loss is the last logged interval at step 340 of the 341-step run, not a whole-run average.
The larger window has lower logged loss than the old window but higher loss than full layers
in each seed. All arms decide 100% of prompts, with no parse failures or refusals. Agreement
uses each model's own greedy operating point, rather than a matched rate; it remains a proxy.
The matched window's agreement differences are +0.129, −0.319 and +0.242 pp versus full.

The merge diagnostic is the projection retained along the intended adapter update:
`sum((W_after - W_before) * D) / sum(D * D)`, with `D = B @ A * scaling` computed in float32
and the merge performed in the base dtype; before/after weights are promoted to float32
to measure the change. It describes the last adapter in
the chain, not an energy fraction or a count of changed weights. The retained projection
is about 99.0% for r163, versus about 92% for full-layer r32 and 96% for old-window r32.
This difference in bf16 merge retention, alongside rank and
optimization geometry, prevents attributing the change solely to parameter count. These are
served merged-checkpoint outcomes. Source manifests, adapter settings, training-state snapshots,
quality records and hashes are indexed in {link('reports/poster_followup/window_budget_guardrails.json')}.

Protocol: {link('docs/POSTER_FOLLOWUP_EXPERIMENTS.md')}. Completed source tables, parameter counts,
bootstrap specification and hashes: {link('reports/poster_followup/window_budget_summary.json')}.
Regenerate the summary and poster macros with {link('poster/latex/window_data.py')}.
Per-seed comparisons and completion markers are under `reports/experiment_followup/window_seed42`,
`window_seed43` and `window_seed44`; accepted jobs are in {link('reports/experiment_followup/submissions.json')}.

## NVIDIA acceleration, timing scope and cost

The platform is CSCS Alps, with four NVIDIA GH200 GPUs per node. The recorded device name is
`NVIDIA GH200 120GB`, consistent with {link('docs/ACCEL_BENCH.md')}. The poster therefore omits
the earlier unsupported `96 GB HBM3` label. These logs did not measure total device memory,
so no additional physical-capacity or HBM-generation claim is inferred from the name.
The measured software stack uses vLLM 0.31, TRL 1.15 / PEFT and PyTorch DDP in bf16.
The inference benchmark has {m['AuditBenchPrompts']} prompts per run and {m['AuditBenchRepeats']} repeats per condition.
The three bars are per-run medians and measure generation only; startup is excluded.
With two observations, each median also equals the arithmetic mean.

{table(['Backend', 'Median generation time per run', 'Speed relative to vLLM'], [
    ['HF batch 8', m['AuditHFSmallSeconds'] + ' s', m['AuditSpeedSmall'] + '×'],
    ['HF batch 32', m['AuditHFLargeSeconds'] + ' s', m['AuditSpeedLarge'] + '×'],
    ['vLLM', m['AuditVLLMSeconds'] + ' s', '1×']])}

HF/vLLM decision agreement is 99.68%–99.78%. For the full 31,050-prompt generation workload,
the medians across four repeats are {m['AuditWallS']} s including startup and writeout,
and {m['AuditKWh']} kWh of GPU energy per run, not energy summed over the four runs.
Its {m['AuditReqPerS']} requests/s measures generation only, about {m['AuditGenS']} s, not end-to-end
throughput. It is also a per-run median. This has {m['FullAuditRepeats']} repeats and controlled condition ordering.
Energy is the difference between NVML device total-energy counters across the complete cost
step, summed over used devices. It includes loading and writing, measures GPU device energy,
and does not measure node/CPU energy or isolate energy attributable to one process.
It replaces the old six-step GRPO timing line, which was not a rollout-only measurement.

The DDP comparison uses reviewed one-GPU repeat job 5238558 and four-GPU job 5236800,
with the same global batch 32 and 40 training steps.

{table(['Timing and GPU-hour basis', '1 GPU', '4 GPUs', 'Wall speed-up', 'GPU-hours, 4 vs 1'], [
    ['Training only', m['DDPOneSeconds'] + ' s', m['DDPFourSeconds'] + ' s', m['DDPTrainingSpeedup'] + '×', m['DDPTrainingCostDelta'] + '%'],
    ['Whole cost step, startup included', m['DDPOneWallSeconds'] + ' s', m['DDPFourWallSeconds'] + ' s', m['DDPEndToEndSpeedup'] + '×', m['DDPCostDelta'] + '%']])}

The repeat ran on an exclusive node. Steps 6–40 stayed near 50 s; SM clock stayed at 1,980 MHz,
power was about 190 W of a 900 W limit, temperature 26–27°C, and throttle counters did not change.
GPU utilization was 17%–21%. Missing `causal_conv1d` and microbatch size one are possible causes
of low utilization; neither contribution was isolated. The first run's slowdown remains unexplained.
The original registration used an absolute 5% phase difference. The accepted one-sided
no-slowdown reading is post hoc and recorded in {link('poster/latex/ddp_review.json')}.
Ledger row 4 and the full rerun account are in {link('docs/ACCEL_BENCH.md')}.

Actual selected-pair metadata, rather than the requested cap, defines the 64-pair patch cost cohort.
It contains {m['PatchRunsSixtyFour']} measured runs, each with 64 pairs. Runtime and used-GPU-hours
are medians across those runs: {m['PatchStepMin']} min and {m['PatchStepGPUh']} GPU-h per model × contrast.
The {m['PatchPeakGiB']} GiB value is the maximum of their sampled device-memory peaks, not a median peak.
Full 31,050-prompt margin scoring is a separate pass: median {m['MarginsStepMin']} minutes across
11 runs and maximum measured peak {m['MarginsPeakGiB']} GiB. Memory peaks are sampled NVML
device-used-memory maxima and include vLLM reservation. Timing/energy collection is documented
in {link('src/hiring_bias_mitigation/utils/cost.py')} and {link('docs/COST.md')}.
Each of the patching and margin-scoring passes fits separately on one GH200 without CPU offload.
The source snapshot totals {m['ProjectGPUh']} **allocated** GPU-h and {m['ProjectKWh']} kWh measured
GPU energy. It excludes follow-up jobs submitted after that snapshot.
All source hashes, selected cost records, excluded-pilot logic and timing scopes are in
{link('reports/poster_followup/compute_summary.json')} and {link('poster/latex/compute_data.py')}.

## References, code and data

The poster includes numbered references and printed project/data addresses. Full citations
and detailed controls are also collected here.

- [1] [Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021](https://arxiv.org/abs/2106.09685).
- [2] [Rafailov et al., Direct Preference Optimization, 2023](https://arxiv.org/abs/2305.18290).
- [3] [Shao et al., DeepSeekMath, introducing GRPO, 2024](https://arxiv.org/abs/2402.03300).
- [4] [Hase et al., Does Localization Inform Editing?, 2023](https://arxiv.org/abs/2301.04213).
- [5] [Zhang and Nanda, Towards Best Practices of Activation Patching, 2024](https://arxiv.org/abs/2309.16042).
- [Project code](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at).
- [Hiring-bias-mitigation-v3 dataset](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-v3).

## Reproduction

1. Run `scripts/poster_audit_followup.py` in the project environment with the canonical raw outputs.
2. Run `poster/latex/latex_data.py --compute-only` to refresh the measured compute block.
3. Run `poster/latex/plot_revision.py` to regenerate the two vector figures.
4. Run `poster/latex/window_data.py` to validate and summarize the three completed budget controls.
5. Run `poster/revision_details.py`, then compile `poster/latex/main.tex` with pdfLaTeX twice.

The poster uses vector charts and embedded fonts. The author and affiliation line remains
an explicit placeholder until the author order is agreed.
"""
    path.write_text(text)


if __name__ == "__main__":
    write()
