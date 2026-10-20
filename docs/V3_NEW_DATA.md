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

The round also stops at the node-hour budget the user sets for the full run.

### Compute

Measured rates on one GH200 node (TP 4), at Djinni's prompt length: 23.3 draws/s for the vote (job 5073453) and 22.4 prompts/s for a variant pass (job 5103710). Loading the teacher adds about 5 minutes per job.

Per 1,000 pairs, that is about 0.10 node-h of votes, 0.15 of invariant and 0.15 of biased, or 0.40 node-h in total. Kaggle x LinkedIn prompts are about twice as long, so their rate is an assumption until the pilot measures it.

- **Pilot.** One job runs every pass on part 000 of each source. Its readouts are checked against the stop conditions before any further part.
- **Full run.** The remaining parts, in jobs of up to 12 hours, one at a time.

### Splits and assembly

No model is trained or selected, and the benchmark is not read.

- **Split.** A seeded 5% of each source's candidates go to validation. No new candidate is a V1 candidate, so SFT-v3's validation stays as built.
- **Balancing.** SFT rows are balanced with `balance_decisions` over the new rows only, so SFT-v3's rows are untouched.
- **Outputs.**
  - `additional_{sft,dpo,kto}_{train,validation}.parquet`: the new rows alone.
  - `sft_{train,validation}.parquet`: `sft_v3_swap` row for row, with the new rows after it. A data config can point at the directory with `data_kind: sft`.

### V1's pairs in the DPO set, and the 10k view

SFT-v3 built SFT data only. To reach the user's target of about 10,000 DPO pairs, `--stage v1_dpo` builds V1's pairs' DPO rows on the vote anchors, from v1's raw passes and on CPU:

- **chosen** is SFT-v3's: v1's invariant targets on the 2,800 unchanged pairs, and job 5103710's regenerated targets on the 192 changed pairs;
- **rejected** is v1's biased generations (`data/raw/en_biased.parquet` of the published dataset, at the revision `semisynthetic-v1-hub` records, sha256 8ef524e6...). `BIASED_PROMPT` does not quote the anchor, so those generations stand for the changed pairs too. Only `filter_biased`'s flip check reads the anchor, and it is re-run against the vote's;
- the split is SFT-v3's: a candidate in `sft_v3_swap`'s validation is in validation.

`--stage assemble` then writes:

- `dpo_*` and `kto_*`: V1's pairs and the new pairs together;
- `dpo_capped_*`: at most 2 rows per pool pair, rows of different protected groups first, then as many hire as reject rows.

The second is the view meant for training. A pair's 12 variants share one CV and one JD, so without the cap the set would be thousands of near-duplicates of a few hundred pairs on the hire side.

Sizing, from v1's English yields (`dpo_*.parquet`: 84.6% of hire-anchor variants and 50.5% of reject-anchor variants become DPO rows) and the vote's 21% hire share: 10,000 capped and balanced rows need about 2,500 hire pairs. V1's pairs give 616. The remainder, about 1,900 hire pairs, needs about 9,000 new pool pairs. Hence the first run's scope: `djinni_new` 000-002, `rjdb` 000-001 and `kaggle_linkedin` 000-001, 9,034 pairs (job 5104067).

### Known caveat: the DPO negatives

These DPO pairs are v1's kind: `rejected` is the teacher asked to let the attribute drive the verdict. In v1, 91% of those named the attribute and 78% kept the chosen verdict, and DPO separated the pairs by wording within 50 steps (`generation/negatives_v2.py` docstring). The project's best DPO set since then, r2em, takes `rejected` from the SFT student's own attribute-tipped mistakes, judged by the teacher (`docs/V2_NEGATIVES_DPO_DATA.md`). Those negatives need the SFT model DPO starts from, so they can only be drawn after SFT-v3 is trained. The anchors and invariant targets made here are the inputs that on-policy build needs.

## Result

Not run yet.
