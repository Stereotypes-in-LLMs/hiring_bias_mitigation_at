"""Rebuild the LaTeX poster's QR evidence page from completed result tables."""
from __future__ import annotations

import csv
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
The tables below use completed measurements. New window-rank and mechanism-seed experiments are
registered separately and are not evidence for this version.

## Scope and interpretation

The development benchmark contains 150 candidates, 450 CV–job pairs, 2,700 attribute sets and
31,050 prompts. Each set varies identity cues while fixing qualifications, job, attribute family
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
Sources: {link('reports/final/final_table.csv')}, {link('reports/final/curves.csv')}.
The family × cue-style breakdown is in {link('reports/poster_followup/family_cue.csv')}.
It uses global matched-rate cuts before stratification, not a new threshold for each subgroup.

## Residual instability and tail sensitivity

The poster illustrates the 43/44 seed pair at each arm's own rate. All pairs are shown here.
The control holds SFT-v3 scores fixed and moves its cut to each arm's hire rate. A low actual
overlap alone cannot separate a recipe change from a threshold change.

{table(['Recipe', 'Seeds', 'Actual Jaccard', 'SFT threshold control'], own)}

Source: {link('reports/seed_overlap/overlap.csv')}. A second analysis fixes every arm at
the same 22.512% hire rate. Fractional tied-set membership uses sum(min weights)/sum(max weights).
The SFT control is then identical across seeds and equals one. Do not mix actual overlaps from
this table with the own-rate controls above.

{table(['Recipe', 'Seeds', 'Common-rate weighted Jaccard', 'Same-cut SFT control'], common)}

Source: {link('reports/poster_followup/common_rate_overlap.csv')}.
The 35 pairs near SFT's cut are 7.8% of the 450 pairs. The share of remaining instability on
these pairs depends on rate convention: the fixed-rate values are recorded per seed in
{link('reports/poster_followup/common_rate_seeds.csv')}.

The tail plot uses one selected checkpoint per method and all 450 pairs for each contrast.
Its p50 and p99 are ratios of quantiles of absolute attribute-induced margin differences,
relative to SFT-v3. The connecting line spans four contrasts; it is not a confidence interval.

{table(['Recipe', 'Contrast', 'Pairs', 'p50 ratio', 'p99 ratio'], tails)}

Sources: the `sensitivity.csv` files under `reports/interp/distill/` and
{link('reports/poster_followup/tail_sensitivity.csv')}. This behavioral tail result is not
established as a consequence of the patched head changes.

## What the interventions establish

The Base-to-SFT panels select pairs by Base effect. The post-SFT comparisons select on SFT-v3
effect, with 64 pairs per contrast except gender, where 45 pairs pass selection. "Same pairs"
applies within a comparison, not across those panels. The Base-selected combat-implicit
intervention at L15.H11 removes about 0.66 logits per pair in Base and 0.13 after SFT.

Selected-pair effects do not estimate the full benchmark effect. In particular, gender's
selected-pair reduction is about 72%, while the full 450-pair reduction is about 25%.
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

The original window-only LoRA has 11.4M trainable parameters versus 58.2M for full-layer r32,
and ends at higher loss. Its negative result is confounded by capacity and optimization.
The registered r163 window has 58.085M parameters versus 58.196M for full-layer r32, within 0.2%.
Pending results cannot yet establish whether localization guides editing.
Protocol and job records: {link('docs/POSTER_FOLLOWUP_EXPERIMENTS.md')} and
{link('reports/experiment_followup/submissions.json')}.

## NVIDIA acceleration, timing scope and cost

The platform is CSCS Alps, with four NVIDIA GH200 GPUs per node and 96 GiB HBM3 per GPU.
The inference benchmark has {m['AuditBenchPrompts']} prompts and {m['AuditBenchRepeats']} repeats per condition.
The three bars measure generation only; startup is excluded.

{table(['Backend', 'Generation time', 'Speed relative to vLLM'], [
    ['HF batch 8', m['AuditHFSmallSeconds'] + ' s', m['AuditSpeedSmall'] + '×'],
    ['HF batch 32', m['AuditHFLargeSeconds'] + ' s', m['AuditSpeedLarge'] + '×'],
    ['vLLM', m['AuditVLLMSeconds'] + ' s', '1×']])}

HF/vLLM decision agreement is 99.68%–99.78%. The full 31,050-prompt generation workload takes
{m['AuditWallS']} s including startup and writeout, with {m['AuditKWh']} kWh measured GPU energy.
Its {m['AuditReqPerS']} requests/s measures generation only, about {m['AuditGenS']} s, not end-to-end
throughput. This has {m['FullAuditRepeats']} repeats and controlled condition ordering.
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
It contains {m['PatchRunsSixtyFour']} measured runs, a median {m['PatchStepMin']} min and
{m['PatchStepGPUh']} used GPU-h per model × contrast, and peak {m['PatchPeakGiB']} GiB.
Full 31,050-prompt margin scoring is a separate {m['MarginsStepMin']}-minute, {m['MarginsPeakGiB']}-GiB pass.
The source snapshot totals {m['ProjectGPUh']} **allocated** GPU-h and {m['ProjectKWh']} kWh measured
GPU energy. It excludes follow-up jobs submitted after that snapshot.
All source hashes, selected cost records, excluded-pilot logic and timing scopes are in
{link('reports/poster_followup/compute_summary.json')} and {link('poster/latex/compute_data.py')}.

## References, code and data

The poster keeps short scope notes beside each figure. Full citations and detailed controls
are collected here so they remain readable rather than forming a small-text footer.

- [Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021](https://arxiv.org/abs/2106.09685).
- [Rafailov et al., Direct Preference Optimization, 2023](https://arxiv.org/abs/2305.18290).
- [Shao et al., DeepSeekMath, introducing GRPO, 2024](https://arxiv.org/abs/2402.03300).
- [Hase et al., Does Localization Inform Editing?, 2023](https://arxiv.org/abs/2301.04213).
- [Zhang and Nanda, Towards Best Practices of Activation Patching, 2024](https://arxiv.org/abs/2309.16042).
- [Project code](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at).
- [Hiring-bias-mitigation-v3 dataset](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-v3).

## Reproduction

1. Run `scripts/poster_audit_followup.py` in the project environment with the canonical raw outputs.
2. Run `poster/latex/latex_data.py --compute-only` to refresh the measured compute block.
3. Run `poster/latex/plot_revision.py` to regenerate the two vector figures.
4. Run `poster/revision_details.py`, then compile `poster/latex/main.tex` with pdfLaTeX twice.

The poster uses vector charts and embedded fonts. Publication still requires the author and
affiliation information, which is intentionally omitted from this revision.
"""
    path.write_text(text)


if __name__ == "__main__":
    write()
