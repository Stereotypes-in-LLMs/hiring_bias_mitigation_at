# Visual explainers for the poster

Two standalone, editable figure candidates. The current poster and Marta's
activation-patching illustration are unchanged.

## Same CV, different cues

`same-cv-audit` turns the existing Sales Manager example into an input-to-output
diagram. The CV has 4.5 years in sales roles; the job requires at least 3 years.
The displayed female/she-her and male/he-him variants receive different greedy
Base decisions. SFT-v3 hires both. Across all 20 gender-cue variants for this
pair, Base hires 15 and rejects 5; SFT-v3 hires all 20.

Tiles encode counts, not identity ordering. This is an illustrative example,
not a population effect estimate. Agreement across cues does not establish
correctness or fairness. The arrows show evaluation inputs, not training data.

Suggested use: replace the left-column prose example with a compact vertical
adaptation. Retain the benchmark size, training-method definitions and
contributions elsewhere. Do not simply shrink this wide figure into the narrow
column; reflow it to preserve label size.

## Effect size and decision changes

`effect-versus-decision` uses measured endpoints from one DPO seed 42 checkpoint,
the implicit combat/civilian contrast, and the served vLLM execution path.
Scores and both thresholds come from that same path. Filled points are combat
cues; open points are civilian cues.
These are calibrated decisions, not the model's raw greedy outputs. "Large"
and "smaller" compare absolute logit effects only. Their empirical percentile
effects have a different ordering; no rank-effect claim is implied here.

- Common target: 6,990/31,050 prompts hired, 22.512077%; cut 7.125.
- High-rate target: 10,868/31,050 prompts hired, 35.001610%; cut -33.09375.
- Large-effect pair `216583fef5661a4c779b`: endpoints -7.125 and 24.9375.
  Its decisions split at the common cut; both are hire at the high-rate cut.
- Smaller-effect pair `bb4540e0134d7fdf6141`: endpoints -32.015625 and -33.578125.
  Both are reject at the common cut; decisions split at the high-rate cut.

No displayed endpoint equals either threshold, so tie interpolation does not
affect these decisions. The upper rows share one unbroken linear scale. The
lower panel is an explicitly labeled zoom of the small pair. Rounded labels
do not replace the exact values in `measured-examples.json`.

Selection is deliberate, not random: take the largest top-1%-effect pair that
crosses the common cut and is entirely above the high-rate cut; then take the
largest-effect pair that strictly crosses the high-rate cut, within the same
seed and contrast. These examples illustrate threshold geometry. They do not
estimate prevalence or explain the full 2,700-set instability curve.

Suggested use: a compact explanatory inset next to the hire-rate curve or the
residual-risk section. Retain the measured population curve and its uncertainty
labels. This figure does not replace those results.

## Reproduce and edit

From the repository root, run:

```sh
python poster/latex/draw_visual_explainers.py
```

Requires matplotlib. The script reads the existing case macros and completed
CPU analysis tables, asserts the selected examples, and writes PNG, PDF, SVG,
plus a JSON source manifest. It does not run models or alter `main.tex`.
SVG text stays editable. PDFs retain vector graphics and embedded TrueType
fonts. The palette is the poster's existing blue, green and gray.
