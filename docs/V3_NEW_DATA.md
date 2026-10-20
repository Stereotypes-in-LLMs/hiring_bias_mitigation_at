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

Not run yet.
