# V3 new data: the SFT-v3 pipeline on pairs V1 never used

SFT-v3 (`docs/SFT_V3_ANCHORS.md`, branch `xinyang/sft-v3-vote-anchors`) changed two things in v1's construction: the 8-draw vote's majority became the anchor, and every filter uses the whole-word `metrics.mentions_attribute` (9e02ae2). Its data sets, `sft_v3_swap` and `sft_v3_rebalanced`, cover V1's 3,000 English Djinni pairs. This round runs the same construction on English pairs V1 never used: the rest of Djinni and the external sources of `docs/EXTERNAL_DATA.md`. The new rows extend SFT-v3's sets and do not replace them. Training is not part of this round; the user trains SFT-v3 and DPO on the result.

The code is `scripts/v3_additional_data.py`, `generation/vote.py` and `configs/generation/v3_additional_data.yaml`, run through `scripts/cscs/v3_additional_data.sbatch`. Outputs go to `artifacts/semisynthetic-v3/additional/` under `$HBM_OUTPUT_ROOT`.

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
