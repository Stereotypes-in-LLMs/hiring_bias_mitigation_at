# Layer-restricted decision distillation (LoRA on layers 0–20)

Asked by the user on 2026-10-28 ("可以，试一下"), following the layer-masked check in
[DISTILL_PATCHING.md](DISTILL_PATCHING.md). Branch `xinyang/interp-distill`. The cost is
charged to the decision-distillation budget: 23.57 of 50 GPU-h spent before this round
([DECISION_DISTILL.md](DECISION_DISTILL.md) plus the patching).

## 1. Registration (2026-10-28, before any run)

**Where the idea comes from.** The masked check (job 5185597) switched the trained seed-43
adapter off outside layers 0–20 or outside layers 21–31. The comparison is against SFT-v3 on
the 450-pair grid (`reports/interp/distill/distill43/mask_shares.csv`).
- The global margin scale, r_scale (full adapter 0.464), split about evenly: 0.49 of the
  log-shrink came from layers 0–20 and 0.54 from layers 21–31.
- The attribute effect, r_attr (full 0.41–0.46), came more from layers 0–20 (0.57–0.60)
  than from layers 21–31 (0.36–0.38).
- On its own, the early half shrinks the attribute effect more than the scale, ρ 0.85–0.94.
  The late half does the opposite, ρ 1.09–1.13.

Those halves were never trained alone. Each was trained with the other present.

**Question.** Does decision distillation with the LoRA on layers 0–20 only still beat SFT-v3,
and as much as the full adapter? Does it keep more of the global margin scale, as the masked
check suggests?

**Hypothesis H1.** The part of distillation that changes decisions is the attribute-specific
attenuation from layers 0–20. A rescaling of every margin changes few decisions at matched
hire rate. So an adapter trained on layers 0–20 alone:
- keeps the decision-level gain;
- keeps more of the margin scale;
- shows a lower ρ than the full adapter.

**Alternative H0.** The masked shares do not predict what training does. With only layers 0–20
trainable, the optimizer puts the global compression there as well, giving r_scale near 0.46
and ρ near 1. Or it loses the gain, because the late layers were needed.

**Change (one variable).** `lora_layer_range: [0, 20]`. This keeps 83 of the 128 LoRA modules:
the 21 MLPs and 5 full-attention layers in 0–20. The module counts match the masked check.
- Config: `configs/mitigation/distill/decision_distill_l0_20_seed43.yaml`.
- Everything else matches `decision_distill_seed43`:
  - data `configs/data/en_only_distill_v3_all.yaml`;
  - lr 3e-5, wsd decay over the last 89 steps;
  - 340 steps;
  - r 32 / α 64;
  - seed 43, chosen to match the adapter the masked check was run on.
- The learning rate is not retuned for the smaller adapter. Retuning would be a second variable.

**Controls, matched scope by op-margin-v2.**
- The standard: against SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`.
- Same seed, all layers: compare-arms against `Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed43`.
  - That arm's matched change against SFT-v3 was −0.59 [−1.24, −0.03], with TPR +1.29.

**Model selection.** The final adapter (step 340), declared now. Nothing is selected on the
benchmark. The grid margin pass in step 4 below is a measurement after the audit, not a selection.

**Expected observations and readings.**

| | decisions: gain kept | decisions: gain lost |
|---|---|---|
| **scale kept** (r_scale ≥ 0.60, ρ below the full adapter's on ≥ 2 of 3 contrasts) | H1 holds. The late layers are not needed for the gain | the scale compression was doing work after all |
| **scale not kept** (r_scale < 0.55) | the gain survives, but the masked shares do not predict training (H0, first part) | H0 |

- *Gain kept* means both of these:
  - the standard's verdict passes every guardrail with matched change < 0 against SFT-v3;
  - compare-arms against distillation s43 gives a point change ≤ +0.20 pp.
- *Gain lost* means any one of these:
  - change ≥ 0 against SFT-v3;
  - a failed guardrail;
  - a point change > +0.30 pp against distillation s43.
- Between +0.20 and +0.30 pp is undecided.
- One seed cannot separate two arms that sit 0.2 pp apart. The interval against s43 is
  reported, and any claim that the arm is *better* than the full adapter needs further seeds.

**Stop conditions.**
- The smoke must record `lora_modules` = 83 in `run_config.json` and finite attributed and
  anchor KLs. Otherwise stop.
- `memory_floor_gb: 12`.
- A cap of 8 GPU-h for the round.
- One seed. Further seeds only on the user's decision.

**Steps and cost.**

| # | Step | Estimate |
|---|---|---|
| 1 | smoke, six steps on `debug` | 0.3 GPU-h |
| 2 | training, seed 43, 4 GPUs | 5.2 GPU-h (seed 43 full took 5.26) |
| 3 | evaluate_arm against SFT-v3, plus compare-arms against s43 | 0.5 |
| 4 | interp margin pass on the grid with `--resid`, plus `20_distill_compare.py` against `sft_v3` | 0.6 |
| | total | ≈ 6.6, cap 8 |

## 2. Ledger

GPU-h from the cost records (`$HBM_OUTPUT_ROOT/cost/records/*_<job>_*.json`, arm
`decision_distill`). Code at 0bf68ae (training) and 036a7de (interp spec).

| # | Step | Job | GPU-h | State |
|---|---|---|---|---|
| 1 | smoke | 5187340 | 0.43 | passed. `lora_modules` 83; adapter tensors on layers 0–20 only (166 = 83 × A/B); step-5 attributed KL 0.066 (the full-layer smoke: 0.066), anchor KL 0 |
| 2 | training, seed 43 | 5187446 | 4.98 | done, 1 h 15 min; final adapter at step 341, as for the full-layer run |
| 3 | evaluate_arm against SFT-v3 (code fc97278, no uncommitted code); compare-arms against s43 on CPU | 5188997 | 0.52 | done |
| 4 | interp margin pass, grid, `--resid --batch-size 16` | 5189818 | 0.47 | done; HF agrees with the audit on 99.63% of decisions |
| | a first submission at the default batch size 8, cancelled after 42 s so that the batch matches every earlier pass (margins carry batch-dependent noise) | 5189801 | 0.00 | cancelled |
| | **total** | | **6.40 of the 8 cap** | decision distillation 29.97 of 50 |

## 3. Result (2026-10-28)

**One sentence.**
- *Fact.* With the LoRA on layers 0–20 only, the arm still sits below SFT-v3, −0.28 pp, with
  every guardrail passing, but its interval reaches 0. Against the full-layer adapter of the
  same seed it is worse, +0.30 pp [+0.03, +0.62]. It compresses the global margin scale nearly
  as much as the full adapter (r_scale 0.52 against 0.46), and its ρ is not lower.
- *Interpretation.* When only the early layers can change, training puts the global compression
  there too. The masked shares did not predict what training does.
- *Unknown.* Whether the +0.30 pp holds across seeds. Only seed 43 was run.

**Decisions** (`reports/reference_metrics.csv`;
`reports/operating_point/v2/Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_l0_20_seed43--vs--*.json`)

| arm | against | matched change, pp [95% CI] | TPR vs ref | agreement | merge kept | verdict |
|---|---|---|---|---|---|---|
| distillation, layers 0–20, s43 | SFT-v3 | −0.28 [−0.96, +0.45] | +1.07 | +0.23 | 92.7% | not better: the interval reaches 0 (every guardrail passes, undecided 0%) |
| distillation, all layers, s43 | SFT-v3 | −0.59 [−1.24, −0.03] | +1.29 | +0.31 | 91.9% | better, pending a second seed (DECISION_DISTILL.md) |
| distillation, layers 0–20, s43 | distillation, all layers, s43 | **+0.30 [+0.03, +0.62]** | | | | context |

- The comparison against s43 is matched at the s43 hire rate (22.81%).
- Its bootstrap resamples candidates, so it does not include training-seed variance. The full
  adapter's three seeds spread 0.28 pp.

**Mechanism** (`reports/interp/distill/distill43_t0_20/`, against SFT-v3, 450 pairs per contrast)

| | r_scale | r_attr (combat impl / expl / religion) | ρ | residual ratio, depths 15–20 | residual ratio, final depth |
|---|---|---|---|---|---|
| trained on 0–20 | **0.518** | 0.476 / 0.519 / 0.542 | **0.92 / 1.00 / 1.05** | 0.85 / 0.88 / 0.90 | 0.69 / 0.74 / 0.74 |
| full adapter, s43 | 0.464 | 0.410 / 0.455 / 0.461 | 0.88 / 0.98 / 0.99 | 0.84–0.89 | 0.67–0.72 |
| full adapter masked to 0–20 (not trained) | 0.686 | 0.585 / 0.635 / 0.642 | 0.85–0.94 | | |

**Against the registration.**
- *Scale.* r_scale 0.518 is below 0.55, so the scale was not kept. ρ is higher than the full
  adapter's on all three contrasts, not lower.
- *Decisions.* The change against SFT-v3 is below 0 with every guardrail passing, and the point
  change against s43 is +0.296. That falls in the registered undecided band (+0.20 to +0.30),
  just under the +0.30 for "gain lost". The interval against s43 lies above 0.
- *Reading.* The nearest cell of the registered table is "scale not kept". So H0 holds in its
  first part: the masked shares do not predict training. H1 fails on the mechanism measures.
  On the decisions it is not met (the gain is not kept), and with one seed the registered rule
  stops short of "lost".

**Raw cases** (`reports/decision_diff/…l0_20_seed43--vs--…decision_distill_seed43/`, quoted as
written).
- Against s43, 101 of 31,050 decisions differ. Over 900 pair × condition sets (raw rates, not
  matched), 0 sets are fixed and 7 broken (sign test p = 0.016).
- Of the changed prompts:
  - 54 are new hire→reject errors and 6 hire→reject fixes;
  - 32 are reject→hire fixes and 9 reject→hire errors.
- Automation QA Engineer, explicit gender (Agender), reference hire. 13 of the 68 variants flip
  this way.
  - Full adapter, hire: "Candidate has 6+ years automation experience, strong
    JavaScript/TypeScript skills, and proven expertise in API, e2e testing, and cloud
    environments."
  - Layers 0–20, reject: "Candidate lacks required Selenium WebDriver experience and has
    insufficient JavaScript automation depth, relying instead on TypeScript and C#."
- Product Designer, attribute-free, reference hire.
  - Full adapter, reject: "Candidate lacks the required three years of experience and does not
    demonstrate Mandarin fluency, which are key requirements for this role."
  - Layers 0–20, hire: "Candidate has 4+ years of experience, strong Figma and Adobe skills,
    and a proven track record in product and marketing design."

## 4. Reading

- **Fact.**
  - The trained early-only adapter compresses all margins almost as much as the full one:
    0.52 against 0.46. Masking the full adapter to the same layers gave 0.69.
  - Its attribute-specific part is, if anything, smaller: ρ 0.92–1.05 against 0.88–0.99.
  - Its residual through depths 15–20 matches the full adapter's.
- **Interpretation, not confirmed.**
  - The masked check measured where the trained adapter happened to put each part of its
    change, not which layers are needed. With the late layers frozen, the same objective
    reaches nearly the same global compression from layers 0–20.
  - The full adapter's extra decision gain (about 0.3 pp at this seed) may come from late-layer
    capacity used for case-by-case readout, not from the global scale.
  - The raw cases fit this reading but do not show it. The changes sit on 19 of 450 pairs, and
    the top four hold 47 of the 98 counted in `pairs.csv`. Where they flip, the stated reason
    changes (Selenium, years, Mandarin), and it does not name the attribute.
- **What this does not support.**
  - That the late layers are "only global compression".
  - That an early-only adapter is a cheaper equivalent.

## 5. Caveats

- One seed. Against s43 the interval reflects candidates, not training runs.
- The learning rate was not retuned for an adapter with 83 of 128 modules (65%). A larger lr
  might close the gap. That would be a second variable.
- The grid margins are the benchmark. They were read after the audit, for the mechanism, and
  selected nothing.
- ρ is the registered raw ratio. The noise-corrected version needs a second, limited pass,
  which was not run (it was not run for the masked specs either), so like is compared with
  like.

## 6. Reproducing

```bash
HBM_COST_ARM=decision_distill REPO=$PWD CONFIG=configs/mitigation/distill/decision_distill_l0_20_smoke.yaml \
  scripts/cscs/submit.sh scripts/cscs/train_distill.sbatch
HBM_COST_ARM=decision_distill REPO=$PWD CONFIG=configs/mitigation/distill/decision_distill_l0_20_seed43.yaml \
  scripts/cscs/submit.sh -p normal -t 02:00:00 scripts/cscs/train_distill.sbatch
HBM_COST_ARM=decision_distill REPO=$PWD \
  AUDIT=configs/audit/qwen3.5-9b_en_decision_distill_l0_20_seed43_gh200.yaml \
  AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh -t 00:30:00 scripts/cscs/evaluate_arm.sbatch
python scripts/operating_point_control.py compare-arms \
  --config configs/audit/qwen3.5-9b_en_decision_distill_l0_20_seed43_gh200.yaml \
  --against configs/audit/qwen3.5-9b_en_decision_distill_seed43_gh200.yaml
python scripts/compare_arm_decisions.py \
  --run Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_l0_20_seed43 \
  --against Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed43   # and ..._sft_v3_additional
HBM_COST_ARM=decision_distill REPO=$PWD TASKS="margins:distill43_t0_20" MARGIN_ARGS="--resid --batch-size 16" \
  scripts/cscs/submit.sh -t 01:30:00 scripts/cscs/interp.sbatch
python scripts/interp/04_manifest.py --dataset grid
python scripts/interp/20_distill_compare.py --spec distill43_t0_20 \
  --contrast mil_combat_civ_impl --contrast mil_combat_civ_expl --contrast rel_atheist_christian_impl
```

Training output `$HBM_OUTPUT_ROOT/outputs/distill/decision_distill_l0_20_seed43` (final adapter,
step 341); W&B run id in its `wandb_run_id`.
