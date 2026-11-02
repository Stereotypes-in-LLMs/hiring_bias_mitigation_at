# The gender contrast under activation patching (registration)

Registered 2026-11-02, before any job, at the user's request: the poster's case is a gender set
and its mechanistic column covers military and religion contrasts only; one (model, contrast)
costs about 20 minutes on one GH200 ([`ACCEL_BENCH.md`](ACCEL_BENCH.md), section 5 of the
poster), so the gap is cheaper to close than to disclaim. Branch `xinyang/accel-bench`, cost
arm `accel_bench`, method and code as in [`RL_PATCHING.md`](RL_PATCHING.md) and
[`DISTILL_PATCHING.md`](DISTILL_PATCHING.md), unchanged.

## Setup

**Question.** Does the attribute pathway patching finds on the military and religion contrasts
(hub L15.H11, layers 15-20) also carry the gender contrast, in the base model, in SFT-v3 and in
the three post-SFT stages?

**Contrast.** `gen_male_female_impl` (`configs/interp/contrasts.yaml`: Male against Female,
implicit wording), already in the manifest (`outputs/interp/manifest/contrasts-grid-en.parquet`).
45 aligned pairs have |Δm_SFT-v3| ≥ 0.5 logits (66 for the base model), so the arms' runs take
all 45 rather than a sample of 64; the base-selected runs take 64 of 66.

**Arms.** Two jobs of `scripts/cscs/interp.sbatch`:
- `patch:{sft_v3,dpo4880,grpo_r4,distill43}:gen_male_female_impl:sft_v3`, the arms against
  SFT-v3 as in section 2C (pairs chosen on SFT-v3's own effect);
- `patch:{base,sft_v3}:gen_male_female_impl:base`, Base against SFT-v3 as in section 2B (pairs
  chosen on the base model's effect, Marta Samoilenko's convention).

**Read as.** For each arm, `20_distill_compare.py` with the four contrasts: profile r against
SFT-v3, L15.H11's rank by the arm's own AtP, the margin scale and the route rule of section 2C
(same if r ≥ 0.80 and rank ≤ 3). For Base vs SFT-v3: mean |Δm| on the selected pairs and the
hub's real-patching effect, the measures of section 2B.

**Expected.** Either outcome is reported: L15.H11 among the top 3 heads on gender too (the
pathway is shared), or not (the pathway is attribute-specific). Nothing on the poster changes
except the mechanistic column's scope and, if it holds, a gender row in 2B.

**Stop.** Two jobs, about 2 GPU-h allocated (six tasks of about 0.33 GPU-h on one node each,
four and two in parallel); a task that fails twice is reported as failed.

## Ledger

| # | job | what | GPU-h allocated | running total |
|---|---|---|---|---|

## Result

## Reproducing

```bash
HBM_COST_ARM=accel_bench REPO=$PWD \
  TASKS="patch:sft_v3:gen_male_female_impl:sft_v3 patch:dpo4880:gen_male_female_impl:sft_v3 patch:grpo_r4:gen_male_female_impl:sft_v3 patch:distill43:gen_male_female_impl:sft_v3" \
  scripts/cscs/submit.sh -t 01:30:00 --exclude=nid005679,nid006377 scripts/cscs/interp.sbatch
HBM_COST_ARM=accel_bench REPO=$PWD TASKS="patch:base:gen_male_female_impl:base patch:sft_v3:gen_male_female_impl:base" \
  scripts/cscs/submit.sh -t 01:30:00 --exclude=nid005679,nid006377 scripts/cscs/interp.sbatch
```
