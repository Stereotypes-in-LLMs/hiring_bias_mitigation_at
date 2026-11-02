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
| 1 | 5237407 | four tasks on nid005330, 15:57 wall: `patch:{sft_v3,dpo4880,grpo_r4,distill43}:gen_male_female_impl:sft_v3`, 45 pairs each, 194 patches per pair and 36–40 verified heads (+8 random controls); 0.22–0.25 GPU-h measured per task (`logs/xw-interp-5237407-*.log`) | 1.06 | 1.06 |
| 2 | 5237408 | two tasks on nid005627, 17:09 wall: `patch:{base,sft_v3}:gen_male_female_impl:base`, 64 of 66 pairs, 33 verified heads each (+8 controls); 0.21 and 0.27 GPU-h measured (`logs/xw-interp-5237408-*.log`) | 1.14 | 2.21 |

## Result

**Fact, the arms against SFT-v3** (45 pairs chosen on SFT-v3's effect; `20_distill_compare.py`
with the four contrasts, `reports/interp/distill/<spec>/`; the three earlier contrasts' rows are
unchanged to the last digit):

| Arm (patched seed) | Profile r, gender | L15.H11 rank, gender (arm's AtP) | top 3 heads, gender | p99 \|Δm\| ratio, gender | Signature over four contrasts |
|---|---:|---:|---|---:|---|
| DPO_V3_4880 s42 | 0.78 | 1 | L15.H11 L15.H1 L19.H11 | 5.5 | partly later (combat implicit: rank 6) |
| GRPO + decay R4 s42 | 0.95 | 1 | L15.H11 L19.H11 L17.H0 | 3.3 | similar profile |
| Self-distillation s43 | 0.95 | 1 | L15.H11 L15.H1 L19.H11 | 0.51 | similar (attenuated) |

The margin scale does not depend on the contrast (×2.68, ×2.16, ×0.46). The p99 ranges over four
contrasts: DPO 4.5–10.7 and GRPO 2.2–3.8, as before; self-distillation 0.40–0.51 (was 0.40–0.43).

**Fact, Base against SFT-v3** (64 of the 66 pairs with |Δm_base| ≥ 0.5, chosen on Base, Marta
Samoilenko's convention; `reports/interp/patching/{base,sft_v3}-grid-gen_male_female_impl-n64-s42/`):
mean |Δm| 0.72 → 0.20 logits per pair (−72%). L15.H11 by real patching at the suffix tokens,
the measure of section 2B's hub: 0.31 → 0.05 logits per pair. L15.H11 is Base's largest
verified head on these pairs (0.44, summed over the two token classes) but not SFT-v3's: on
them SFT-v3's largest are L27.H5 (0.070), L19.H11 (0.068) and L19.H9 (0.067), L15.H11 0.056.
On the 45 pairs chosen on SFT-v3's own effect, L15.H11 is SFT-v3's largest verified head (0.22).

**Against the registration.** Both outcomes were registered as reportable. L15.H11 leads the
gender contrast in all three arms, so the pathway's hub is shared across the four contrasts,
and the arms' signatures are unchanged by adding it: DPO's later route is a combat-implicit
effect, not a general one.

**Interpretation, not confirmed.** On Base-selected pairs SFT-v3 leaves 0.20 logits, too little
for a stable ranking of heads that each carry under 0.1; the hub's loss of rank there may be
noise at that floor rather than a moved route.

**Unknown.** One seed per arm; 45 pairs for the arms, so the gender profile r has wider noise
than the 64-pair contrasts; the gender explicit contrast was not run.

## Reproducing

```bash
HBM_COST_ARM=accel_bench REPO=$PWD \
  TASKS="patch:sft_v3:gen_male_female_impl:sft_v3 patch:dpo4880:gen_male_female_impl:sft_v3 patch:grpo_r4:gen_male_female_impl:sft_v3 patch:distill43:gen_male_female_impl:sft_v3" \
  scripts/cscs/submit.sh -t 01:30:00 --exclude=nid005679,nid006377 scripts/cscs/interp.sbatch
HBM_COST_ARM=accel_bench REPO=$PWD TASKS="patch:base:gen_male_female_impl:base patch:sft_v3:gen_male_female_impl:base" \
  scripts/cscs/submit.sh -t 01:30:00 --exclude=nid005679,nid006377 scripts/cscs/interp.sbatch
```
