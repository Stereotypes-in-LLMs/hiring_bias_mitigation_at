# V3 new data: the SFT-v3 pipeline on pairs V1 never used

SFT-v3 (`docs/SFT_V3_ANCHORS.md`, branch `xinyang/sft-v3-vote-anchors`) changed two things in v1's construction: the 8-draw vote's majority became the anchor, and every filter uses the whole-word `metrics.mentions_attribute` (9e02ae2). Its data sets, `sft_v3_swap` and `sft_v3_rebalanced`, cover V1's 3,000 English Djinni pairs. This round runs the same construction on English pairs V1 never used: the rest of Djinni and the external sources of `docs/EXTERNAL_DATA.md`. The new rows extend SFT-v3's sets and do not replace them. Training is not part of this round; the user trains SFT-v3 and DPO on the result.

The code is `scripts/v3_additional_data.py`, `generation/vote.py` and `configs/generation/v3_additional_data.yaml`, run through `scripts/cscs/v3_additional_data.sbatch`. Outputs go to `artifacts/semisynthetic-v3/additional/` under `$HBM_OUTPUT_ROOT`.

A second round, at the end of this page, trains SFT-v3 on the result and builds the on-policy part of V3's DPO set from it (user, 2026-10-20).

## Pre-registration

Written on 2026-10-20 at about 11:00 CEST, before any generation, after the user asked for new data from the v3 pipeline ("这是 v3，换了 v1 anchor，调整了 mention_attribute 策略，用这个新管线生成新数据，之后我统一开始训练 v3 SFT，DPO").

### Question

Does SFT-v3's construction give usable SFT and DPO data on pairs beyond V1's? How do the anchors and the yields differ by source?

### The pipeline

Per new pair, v1's construction (`scripts/generate_training_data.py`) with SFT-v3's two changes:

1. **Anchor.** `synth.REFERENCE_PROMPT` drawn 8 times at v1's decoding (temperature 0.7, top_p 0.9, 320 new tokens, thinking off). The majority of decided draws is the anchor, and its feedback is the first majority draw that passes `filters.filter_reference`. `generation/vote.py` ports `sft_v3_anchors.new_anchors` and reproduces job 5073453's 2,992 anchors exactly (`tests/test_vote.py`).
2. **Variants.** v1's grid: military status, gender and religion, 2 attributes each, explicit and implicit, so 12 variants per pair. They come from `synth.build_variant_frame`, seeded per part.
3. **Invariant.** `synth.INVARIANT_PROMPT` with the vote anchor quoted, then `filters.filter_invariant`. These are the SFT targets and DPO's `chosen`.
4. **Biased.** `synth.BIASED_PROMPT`, then `filters.filter_biased`. This is DPO's `rejected`.

Two points differ from SFT-v3 on V1's pairs, both forced by the new pairs:

- **No v1 fallback.** On V1's pairs a tie kept v1's anchor. A new pair has none, so a tie, a pair with no decided draw, and a majority with no usable feedback leave it unanchored and out of every later pass. On job 5073453's draws, this rule would have left 72 of 2,992 pairs (2.4%) unanchored.
- **Per-prompt seeds for the vote.** Job 5073453 sent every pair the shared seed 43. New pairs are drawn with `per_prompt_seed(43, prompt)`, which `VLLMBackend` has used since 7112197. The seed check found the two regimes give nearly the same hire share per draw (0.2147 per prompt against 0.2173 shared; `docs/SFT_V3_ANCHORS.md`). The variant passes use `per_prompt_seed(42, prompt)`, as the vote targets did (job 5103710).

V1's pairs are not drawn again. They keep their SFT-v3 anchors and targets, and enter only so that no new pair repeats a V1 candidate or a V1 (JD, CV) text.

### The new pool

| part prefix | what |
|---|---|
| `djinni_new` | Djinni candidates V1 did not use, one job each, through `pool.match_pairs` (seed 4242) |
| `rjdb` | RJDB train triples, matched CV only, cleaned by `generation/sources.py` |
| `kaggle_linkedin` | Kaggle resumes x LinkedIn JDs, 2 JDs per CV, cleaned and filtered by `generation/sources.py` |

Pair counts per source are in `pool/pool_report.json`.

Each source goes through its own builder's filters: holdout ids, 400-6,000 characters, protected markers, and for the external sources their cleaning and requirement filters. The combined pool then has:

- texts that repeat a V1 pair or an earlier new pair dropped, because a per-prompt seed depends on the text alone and two copies would draw the same samples;
- prompts that cannot fit `max_model_len` 8192 with the invariant prompt's overhead dropped;
- benchmark ids and the external text holdout (`data/external.py`) checked, with matches dropped.

The pool is cut into parts of 1,500 pairs named by source. A job generates the parts it is given.

### Control

- **v1's English sets** (`semisynthetic-v1/generation_report.json`): anchors hire 14.2%, invariant yield 99.5%, biased yield 55.6%.
- **SFT-v3's vote on V1's pairs** (`reports/sft_v3_anchors/pool_votes.md`):
  - per-draw hire share 0.217;
  - anchor hire share 0.206;
  - 82.9% of pairs unanimous;
  - 0 undecided draws;
  - 13.1% of draws in a code fence;
  - the vote targets' invariant yield was 1.000.

### Expected

- On `djinni_new`, the anchor's vote readouts close to SFT-v3's on V1's pairs, since both are Djinni pairs from the same matcher.
- External sources differ in hire share. RJDB's matched CVs were written to fit their JD, so a higher hire share there is expected.
- Invariant yield above 0.95 and biased yield near v1's 0.556 on every source.

### Stop conditions

A source's generation stops after its first part, and what is done is reported, if any of these holds on that part:

- undecided draws above 5%;
- unanchored pairs above 10%;
- invariant yield below 0.80 (the vote targets' floor);
- biased yield below 0.20.

The job enforces them itself (`configs` `stop:`): after each source's first part in a job, `stage_generate` reads the four readouts from that part's files, writes them to `parts/<part>/stop_check.json`, and gives a source that crosses one no further part. The round also stops at the node-hour budget the user sets for the full run.

### Compute

Measured rates on one GH200 node (TP 4), at Djinni's prompt length: 23.3 draws/s for the vote (job 5073453) and 22.4 prompts/s for a variant pass (job 5103710). Loading the teacher adds about 5 minutes per job.

Per 1,000 pairs, that is about 0.10 node-h of votes, 0.15 of invariant and 0.15 of biased, or 0.40 node-h in total. Kaggle x LinkedIn prompts are about twice as long, so their rate is an assumption until the pilot measures it.

- **Pilot.** One job runs every pass on part 000 of each source. Its readouts are checked against the stop conditions before any further part.
- **Full run.** The remaining parts, in jobs of up to 12 hours, one at a time.

### Splits and assembly

Revised at about 13:40 CEST, during job 5104067. The per-source balancing and the exclusions come from the pre-run data-quality review. The DPO outputs come from the addendum below. The rest is as pre-registered.

No model is trained or selected, and the benchmark is not read.

- **Split.** A seeded 5% of each source's candidates go to validation. No new candidate is a V1 candidate, so SFT-v3's validation stays as built.
- **Balancing.** Each pool source's SFT rows are balanced with `balance_decisions` on their own. Pooling them would let RJDB, whose matched CVs were written to fit their JD, supply the hire side and leave the new Djinni rows reject-heavy (data-quality review). SFT-v3's rows are untouched.
- **Exclusions.** Some pairs are dropped at assembly, on their CV, after the teacher has answered them, so the generated parts stay valid (`assemble.exclude`):
  - `djinni_new` CVs that fail the external sources' marker list. That list catches names, third-person pronouns and maternity leave, which V1's list misses and which contradict an injected gender. About 3.3% fail, the same rate as V1's own pairs, which keep V1's filter.
  - RJDB CVs with template residue such as "Details not provided" or "[insert ...]".
- **Outputs.**
  - `additional_sft_{train,validation}.parquet`: the new rows alone.
  - `sft_{train,validation}.parquet`: `sft_v3_swap` row for row, with the new rows after it. A data config points at the directory with `data_kind: sft`.
  - `raw/en_invariant.parquet` and `sft_validation.parquet`: the input of the on-policy DPO build below, in v1's columns.
  - `r2em_v3_{train,validation}.parquet`: from `--stage r2em`, below.
  - `teacher_biased_*`: only with `assemble.teacher_biased_dpo: true`, as a comparison.

### Addendum: DPO is r2em plus on-policy pairs

Written on 2026-10-20 at about 13:40 CEST, during job 5104067, after these readouts and the user's decision.

**What was seen.** The biased teacher never answers hire: 0 of v1's 36,000 English biased generations, by anchor:

- hire anchors: 4,343 reject, 757 refused;
- reject anchors: 26,793 reject, 4,107 refused.

So every `rejected` says reject. A hire-anchor row contrasts hire with reject, and a reject-anchor row contrasts two rejects that differ only in their rationale. On V1's pairs rebuilt on the vote anchors (`--stage v1_dpo`, 19,645 rows), both capped views separate at D3 = 1.000 (`direction_check.surface_report`, balanced AUC, gate 0.65), and D2 fails. These are the negatives r2em replaced.

**The decision.** These DPO pairs are not for training. V3's DPO set is r2em plus new pairs built the r2em way: `rejected` is the student's own attribute-tipped mistake, judged by the teacher, sampled from the SFT model DPO starts from. That model is SFT-v3, so those pairs are drawn after SFT-v3 is trained, with `scripts/generate_onpolicy_full.py` and this directory as `source_artifacts_dir`.

**What changed in the run.** The biased pass is skipped for the parts still to run. Job 5104067 is cancelled after `djinni_new-001`'s invariant pass, before its biased pass, and the remaining parts are resubmitted with `PASSES="votes invariant"`.

- Parts with a biased pass on disk: `djinni_new-000`, `rjdb-000`, `kaggle_linkedin-000` and `kaggle_linkedin-001`. These are kept for comparison only.
- The stop conditions are still read on each source's first part, without the biased yield where that pass did not run.

**r2em on the vote anchors** (`--stage r2em`, `r2em_v3_report.json`). r2em was built on V1's pairs with v1's anchors. Two kinds of rows are dropped:

- 49 train and 20 validation rows whose chosen verdict the vote overturns. All of them chose reject where the vote hires, so DPO would push against the anchor SFT-v3 trains on.
- r2em's `legacy` rows, 42 train and 3 validation, which are v1's biased-teacher negatives.

Kept: 2,271 train and 272 validation rows. Their contrasts are still symmetric: 1,139 hire over reject and 1,132 reject over hire in train.

**Caveat.** r2em's negatives were sampled from SFT v1, and the new on-policy pairs will be sampled from SFT-v3. For a DPO run that starts from SFT-v3, the r2em part is off-policy. That mix is the user's choice, and the per-source readouts should keep the two parts apart.

## Result

**In one sentence.** The v3 pipeline ran on 9,034 new English pairs with no stop condition crossed.

- **Observed.** The pairs come from three sources: `djinni_new` 000-002, `rjdb` 000-001 and `kaggle_linkedin` 000-001. Together they add 54,342 SFT training rows to SFT-v3's swap set.
- **Interpretation.** The vote and the invariant pass behave on the new sources as they do on V1's pairs, apart from the sources' own hire shares.
- **Unknown.** Whether training on these rows helps; nothing was trained.

**Runs.** Both on one GH200 node (TP 4), `normal` partition.

| job | code | parts and passes | elapsed | end |
|---|---|---|---|---|
| 5104067 | 56c8959 | `djinni_new-000`, `rjdb-000`, `kaggle_linkedin-000`/`-001`: votes, invariant, biased; `djinni_new-001`: votes, invariant | 2:15:36 | cancelled on purpose before `djinni_new-001`'s biased pass (addendum) |
| 5105355 | b3fad61 | `rjdb-001`, `djinni_new-002`: votes, invariant (`PASSES="votes invariant"`) | 0:47:50 | completed |

Outputs are under `artifacts/semisynthetic-v3/additional/`: `report.json`, `generation_report.json`, `r2em_v3_report.json` and `parts/<part>/`.

**Compute.** Both jobs had one node, 4 GPUs (`sacct -j 5104067,5105355 -o ElapsedRaw,AllocTRES`):

- 8,136 s and 2,870 s, so 3:03:26 in all: 3.06 node-h, or **12.23 allocated GPU-h**.
- Energy: 16.3 and 5.0 MJ.

By pass, from `parts/<part>/<pass>.meta.json`:

| pass | GPU-h |
|---|---|
| votes | 3.44 |
| invariant | 5.24 |
| biased (not used for training) | 2.76 |
| teacher loading | 0.68 |
| job setup, stop checks and the report | about 0.1 |

This is generation only: no training, and no CPU work on the login node. The cost ledger of `xinyang/cost-ledger` was not merged and recorded nothing.

### The vote, by source (`report.json`, `vote.by`)

| source | pairs | anchored | per-draw hire | anchor hire | unanimous | unstable (at most 5 of 8) | undecided draws | fenced draws |
|---|---|---|---|---|---|---|---|---|
| djinni_new | 4,500 | 4,399 | 0.231 | 0.223 | 0.816 | 0.069 | 0.0000 | 0.132 |
| rjdb | 3,000 | 2,866 | 0.558 | 0.565 | 0.652 | 0.134 | 0.0005 | 0.033 |
| kaggle_linkedin | 1,534 | 1,507 | 0.157 | 0.147 | 0.831 | 0.067 | 0.0000 | 0.285 |
| SFT-v3's vote on V1's pairs (control) | 2,992 | 2,920 without fallback | 0.217 | 0.211 | 0.829 | 0.069 | 0 | 0.131 |

`djinni_new` reads like V1's pairs. RJDB's matched CVs hire 2.5 times as often and split their votes more often. Kaggle x LinkedIn hires least: its pairs are matched on occupation alone, with no experience matching.

### Passes and stop conditions

| source | invariant kept / sent | biased kept / sent (parts with that pass) |
|---|---|---|
| djinni_new | 52,711 / 52,788 (0.9985) | 8,764 / 17,520 (0.5002), `djinni_new-000` |
| rjdb | 34,315 / 34,392 (0.9978) | 8,056 / 17,220 (0.4678), `rjdb-000`; 7,456 refused |
| kaggle_linkedin | 18,032 / 18,084 (0.9971) | 10,304 / 18,084 (0.5698) |

Every first-part stop check came out clear (`parts/<part>/stop_check.json`, collected in `report.json`). The largest unanchored share was 4.6% (`rjdb-001`), against a limit of 10%. The biased teacher answered hire 14 times in 52,824 generations (`biased_decisions_by_source`), which is the one-directional contrast the addendum is about.

**Throughput.** A vote pass took about 510 s per 1,500 pairs (23.5 draws/s). An invariant pass took about 770 s per 17,500 prompts (22.8 prompts/s); Kaggle x LinkedIn took 940 s, its prompts being twice as long. The teacher took about 305 s to load per job.

**Persona** (`persona_by_source`, IT against non-IT roles):

| source | role | pairs | anchor hire share | rationales citing technical skills |
|---|---|---|---|---|
| djinni_new | IT | 2,775 | 0.248 | 0.079 |
| djinni_new | non-IT | 1,624 | 0.182 | 0.188 |
| rjdb | IT | 1,136 | 0.545 | 0.094 |
| rjdb | non-IT | 1,730 | 0.578 | 0.090 |
| kaggle_linkedin | IT | 122 | 0.041 | 0.213 |
| kaggle_linkedin | non-IT | 1,385 | 0.157 | 0.066 |

On RJDB and Kaggle the non-IT roles are not marked down, and technical words go with IT roles. Djinni's non-IT gap is therefore more likely a property of its non-IT pairs than of the "technical recruiter" persona. That is not tested.

### Assembly (`generation_report.json`)

- **Exclusions on CV.** 133 `djinni_new` pairs were dropped for the external marker list, and 88 RJDB pairs for template residue.
- **Balancing, per source** (before, then after):
  - djinni_new: 11,494 hire and 39,656 reject, then 11,494 each;
  - rjdb: 18,734 hire and 14,549 reject, then 14,549 each;
  - kaggle_linkedin: 2,664 hire and 15,368 reject, then 2,664 each.
- **SFT.** `sft_train` has 69,232 rows: `sft_v3_swap`'s 14,890 unchanged, then 54,342 new. The English rows a run reads are 62,761 train and 3,527 validation (`configs/data/en_only_sft_v3_additional.yaml`, checked through `training_common.load_generated_dataset`).
  - New train rows by source: djinni_new 10,822 hire and 10,883 reject; rjdb 13,779 and 13,841; kaggle_linkedin 2,508 and 2,509.
- **On-policy input.** `raw/en_invariant.parquet`: 102,660 rows over 8,555 pairs, in v1's columns.
- **r2em on the vote anchors.** 2,271 train and 272 validation rows (`configs/data/en_only_dpo_v3_r2em.yaml`, `data_kind: r2em_v3`).

## Reading

The new sources give SFT-v3 about seven times its English training rows. Within each source, the targets meet the same filters at the same yields as V1's. The vote and the yields are measured. What the rows do to a model is not.

The hire shares differ by source (0.22, 0.57, 0.15). The per-source balancing keeps each source at 50/50 in SFT, but the mix of sources still changes what the model sees: RJDB is 51% of the new rows. A run on this set should be read per source on the validation split, and at matched hire rate on the benchmark.

## Caveats

- **The DPO negatives.** No DPO set for training is built here. Teacher-biased negatives are ruled out (addendum), and the on-policy pairs need SFT-v3. The biased passes on disk are kept only for comparison: four parts, plus `teacher_biased_v1_dpo_*`.
- **Off-policy r2em.** The r2em part of V3's DPO set was sampled from SFT v1.
- **Ukrainian.** `sft_train` keeps the swap set's 6,471 Ukrainian rows, and every new row is English. The English data config reads English only.
- **V1's own filter.** V1's pairs keep V1's marker filter. About 3.3% of their CVs would fail the external list that the new Djinni rows pass.
- **Seeds.** New anchors use per-prompt seeds, while V1's vote (job 5073453) used one shared seed.
- **The pool is larger than what ran.** It holds 186,970 new pairs; 9,034 were generated, sized for the DPO target.

## Reproducing

```bash
CFG=configs/generation/v3_additional_data.yaml
HBM_OUTPUT_ROOT=$XW python scripts/v3_additional_data.py --config $CFG --stage pool        # CPU, ~15 min
REPO=$PWD PASSES="votes invariant" scripts/cscs/submit.sh -p normal -t 06:00:00 \
    scripts/cscs/v3_additional_data.sbatch djinni_new-000 rjdb-000 kaggle_linkedin-000 ...
HBM_OUTPUT_ROOT=$XW python scripts/v3_additional_data.py --config $CFG --stage r2em        # CPU
HBM_OUTPUT_ROOT=$XW python scripts/v3_additional_data.py --config $CFG --stage assemble    # CPU
```

## Round 2: SFT-v3 and the on-policy part of V3's DPO set

### Pre-registration

Written on 2026-10-20 at about 16:10 CEST. At that point SFT-v3 was not trained and no on-policy sample was drawn. At about 15:45 the user chose DPO's start point: SFT-v3 on all the new data for one epoch ("use all new data train 1 epoch"). No SFT-v3 checkpoint existed under `outputs/sft/`, so this round trains it first. The user asked to stop after this section is committed and before any job is submitted ("stop after commit before run"). Then, before any job, the branch was rebased onto `main` at 6f4b7e9 ("rebase first"), so every job of this round runs with `main`'s cost recorder (`docs/COST.md`).

#### Question

Does r2em's construction work on the new pairs when SFT-v3 is the student? Usable means, for each pool source:

- enough of SFT-v3's own covert, attribute-tipped mistakes on both decision sides to balance them by error mass;
- pairs that pass D1 to D4 and D6 (`docs/DATA_EVALUATION.md`).

The hypothesis is that it works on every source. On djinni_new and kaggle_linkedin, hire anchors are only 22% and 15% of rows, and the k=32 top-up is expected to make their hire side balanceable.

#### The runs

The jobs run one at a time on the `zhang` account, on the `normal` partition, from this worktree (`REPO=$PWD`):

| # | job | config | output |
|---|---|---|---|
| 1 | `train_sft.sbatch` | `configs/mitigation/sft/qwen3.5-9b_en_only_sft_v3_additional_gh200.yaml` | `outputs/sft/qwen3.5-9b_en_only_sft_v3_additional` |
| 2 | `onpolicy_full.sbatch sample` | `configs/generation/negatives_v3_onpolicy_full_gh200.yaml` | `artifacts/semisynthetic-v3/onpolicy-full`: rows, prompts, the merged student, 8 samples per prompt, the selected negatives |
| 3 | `onpolicy_full.sbatch judge` | the same | the teacher's verdicts, candidates, split, the GPU-free direction check |
| 4 | `onpolicy_full.sbatch sample` | `configs/generation/negatives_v3_onpolicy_hire_gh200.yaml` | `artifacts/semisynthetic-v3/onpolicy-hire32`: the hire rows of djinni_new and kaggle_linkedin at 32 samples |
| 5 | `onpolicy_full.sbatch judge` | the same | as job 3 |
| 6 | `build_balanced_dpo.sbatch score` | SFT-v3 | SFT-v3's decision margin on every candidate prompt |
| 7 | direction checks | SFT-v3 and the base | D1 to D7 per pool source, D4 on 600 pairs per source |

The merge, the candidate and build stages and the combine run on the login node, between jobs 5 and 7.

Every job is submitted with `HBM_COST_ARM`, as `docs/COST.md` asks of the v3 runs: `sft_v3` for job 1, and `dpo_v3` for jobs 2 to 7, which DPO's training will share. `reports/cost/by_arm.csv` then adds up each arm.

**SFT-v3.** The recipe is SFT v1's, as the swap arm on `xinyang/sft-v3-vote-anchors` carries it: LoRA r 32 and alpha 64, lr 1e-4 cosine, effective batch 16 as 4 GPUs × 1 × 4. The data is `configs/data/en_only_sft_v3_additional.yaml`: 62,761 English training rows (the swap set's 8,419, then 54,342 new) and 3,527 validation rows. One epoch is 3,923 steps. Besides the data and the single epoch, three settings differ from the swap arm's:

- **`max_seq_len` 3072, not 2048.** At 2048, 629 training rows would be cut at the end, and under a completion-only loss the end is the target. 624 of them are Kaggle × LinkedIn rows (of 5,017) and 5 are djinni_new rows. The longest row is 2,543 tokens (`reports/v3_new_data/sft_token_lengths.json`, from `scripts/sft_token_lengths.py`, which reads the rows through the trainer's own loader and chat template). At batch 1 there is no padding, so every row that fits in 2048 trains as before.
- **No checkpoint selection.** There is no early stopping and no reload of the best checkpoint. The final adapter, after one epoch, is DPO's start point and reference, chosen now.
- **Evaluation and checkpoints.** Every 250 steps the run evaluates on a seeded 1,000 of the validation rows and makes greedy decisions on 64 of them. It also saves a checkpoint every 250 steps and keeps all of them, so a cut job resumes.

**On-policy pairs.** `scripts/generate_onpolicy_full.py` reads `raw/en_invariant.parquet` from `artifacts/semisynthetic-v3/additional`. v1's invariant filter keeps 102,465 rows over 8,549 pairs: djinni_new 51,150 rows, rjdb 33,283 and kaggle_linkedin 18,032.

- **Full run.** 8 samples per prompt: 111,014 prompts, 888,112 samples.
- **Top-up.** 32 samples per prompt on the hire-anchor rows of djinni_new and kaggle_linkedin: 14,158 rows over 1,181 pairs, so 15,339 prompts and 490,848 samples. RJDB's 18,734 hire rows are left out (the user's choice, at about 16:10). RJDB hires 56%, so its hire side is likely not the scarce one, and its rows would have been 57% of the top-up. With them it was 1,140,320 samples.

Everything else is onpolicy-full's: the student's decoding (temperature 1.0, top-p 1.0, 200 new tokens), the selection (`min_free_right` 0.5, length delta at most 10 words), the judge (plausibility 3, the 122B teacher at temperature 0).

Three code changes, each with a test in `tests/test_onpolicy.py`:

- **`val_fraction: 0`** holds out exactly the candidates that `sft_validation.parquet` holds out (`generate_onpolicy_full.validation_ids`). The v1 configs' 0.05 adds a seeded 5% of the split rows' candidates. On v1's pool that draw was v1's own split, so it added nothing. The new pairs were split per pool source, so the same draw would hold out about 5% of the candidates SFT-v3 trained on.
- **`pool_source`.** The judged pairs keep it (`generate_onpolicy_full.carry_row_columns`); `build_pairs` dropped it.
- **A list in `row_filter`** matches any of its values (`onpolicy.filter_rows`). The top-up names its two sources this way.

**Balance, as r2em did.** Each pool source's training pairs are balanced by error mass on their own, with `balanced_dpo.balance_error_mass` at r2em's parameters: tolerance 0.05, `min_keep` 0.5, seed 42 (`v2/round2-error-mass/dpo_round2_report.json`). The margins are SFT-v3's.

- **Candidates.** The full run's pairs with the top-up's merged in (`onpolicy.merge_topup`, the full run's pair first), unbalanced. That is r2em's candidate set: r1 plus the surplus that r1's count balance dropped. There is no legacy pool.
- **Validation.** Balanced by count within each pool source and protected group, the way stage 8 balanced r1's validation, which r2em kept.

That needs code that is not written yet. It will be written before job 6, without changing the rule above:

- a `merge` stage in `generate_onpolicy_full.py` that writes the merged, unbalanced pairs;
- in `build_balanced_dpo.py`, `pool_source` kept through `normalise_pairs` and a `--balance-by pool_source` build; a `NAME` for `build_balanced_dpo.sbatch`;
- a `dpo_v3` stage in `scripts/v3_additional_data.py`, and a job for the per-source checks.

**Combine.** V3's DPO set is r2em_v3 (2,271 training and 272 validation rows) followed by the balanced on-policy pairs, as `dpo_v3_{train,validation}.parquet` beside r2em_v3, read through a new data config. Every row keeps its `pool_source` (r2em_v3's is `djinni_v1`), and a column names the part it came from.

#### Control

- **For the construction:** r2em's own stages, from SFT v1 on V1's pairs.
  - Full run: 5,589 tipped negatives from 35,826 rows (15.6%); the judge kept 4,267 (`v2/onpolicy-full/sample_report.json`, `report.json`).
  - Hire top-up: 2,111 from 5,072 hire rows (41.6%); the judge kept 1,224 (`v2/onpolicy-hire32/report.json`).
  - Before balancing, the error mass was 758.15 on the reject side, over 3,098 pairs, and 106.41 on the hire side, over 1,181 (`dpo_round2_report.json`).
- **For the direction checks:** r2em_v3, checked in the same job under SFT-v3 and the base. r2em's final set has never been direction-checked (`docs/DATA_EVALUATION.md`). This is its first readout, under the student it now serves.
- **For the student:** SFT-v3's own attribute-free hire rate per source, from the full run's samples, since which side a student errs on follows from its operating point. When DPO is evaluated, it is compared with SFT-v3 at matched hire rate (`scripts/operating_point_control.py`). That comparison is not part of this round.

#### Expected

At r2em's rates:

- the full run keeps about 12,200 pairs (102,465 × 0.119);
- the top-up keeps about 3,400 (14,158 × 0.241), and about 1,900 of them are on rows the full run has no pair for (622 of 1,092 in r2em).

Per source, the counts follow the sources' hire shares and the student's errors. Two things differ from r2em, so the rates may be well off, and these numbers are a scale, not a gate:

- SFT-v3 trains one epoch where SFT v1 trained two;
- it sees a different share of each source's rows, since each source was balanced 50/50 on its own.

What would support the hypothesis:

- every source balances by error mass at `min_keep` 0.5;
- on every source, D1 = 1.0, D2 at most 5% lexical and 10% judge allusion, D3 balanced AUC at most 0.65, SFT-v3 preferring `chosen` on 50% to 90% of pairs (D4), and D6 between 0.4 and 0.6 after balancing.

What would count against it: a source whose hire side cannot be balanced even with the top-up, or a source above 0.65 on D3, whose negatives would then carry a wording shortcut.

#### Stop conditions

- **SFT-v3.**
  - `memory_floor_gb` 12 stops the run cleanly with a checkpoint.
  - A job cut by its 10-hour limit is resubmitted once from its checkpoint; a second cut is reported first.
  - The run stops if the loss diverges, or if at any evaluation after the first more than 5% of the greedy decisions on the 64 validation prompts are unreadable.
- **Samples.** After the full run's sample job, a source with tipped negatives on fewer than 2% of its rows is reported before the judge runs. For comparison, the DPO student's round on V1's pool found 409 tipped rows against SFT's 5,614, and its pairs failed D6 at a hire share of 0.013 (`docs/DATA_EVALUATION.md`).
- **Judge.** If the judge keeps under 40% of a run's selected negatives, that is reported before the next job (D7).
- **Balance.** A source that `balance_error_mass` cannot balance at `min_keep` 0.5 stays out and is reported. Neither `min_keep` nor the tolerance is lowered without the user. If RJDB's hire side is what falls short, the remedy is a top-up on its hire rows, at the user's call.
- **Checks.**
  - D1 below 1.0 is a bug: the round stops.
  - A source that fails D2, D3 or D4 is reported before the combine, and enters only if the user says so.
- **Compute.** At most 40 allocated GPU-h for SFT-v3 and 60 for the whole round, as `reports/cost/steps.csv` counts them.

#### Splits

- **SFT-v3** trains on `sft_train`'s English rows and evaluates on `sft_validation`'s. No checkpoint is selected; the final adapter is used.
- **The on-policy pairs** hold out exactly SFT-v3's validation candidates. r2em_v3 keeps r2em's split. The two parts share no candidate, because no new pair repeats a V1 candidate.
- **DPO's checkpoint** is selected later on this set's validation split. The benchmark audit is the final test and is spent once, on the chosen checkpoint. Nothing in this round reads the benchmark.

#### Compute estimate

From past jobs on one GH200 node:

- **SFT.** 47.0 M training tokens. The RFT control (job 5066776) trained 1.83 M tokens in 150 steps: 3.9 s per step at its end, and 855 s for all 150 steps with six evaluations. At those rates, one epoch takes 4 to 6 hours, plus about 25 minutes of evaluation.
- **Sampling.** Job 5044997 drew 310,544 samples in 20 minutes, loading included. At that rate the full run takes about an hour and the top-up about 30 minutes; Kaggle's longer prompts may slow both.
- **Judging.** Job 5045685's judge stage took 13 minutes for 11,178 rationales, about 5 of them loading the teacher (305 s in this round's generation jobs). At r2em's selection rates, the full run has about 32,000 rationales to judge and the top-up about 12,000, so the judge jobs take about 30 and 15 minutes.

That is about 10 node-hours in all, or 40 allocated GPU-hours.
