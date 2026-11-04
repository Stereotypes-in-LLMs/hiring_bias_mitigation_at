# Hiring audit results website

Static research companion for the poster. Site URL:
https://hiring-audit-results.xinyangwu0617.chatgpt.site

The editable site is in `dist/`. Serve that directory with any static HTTP server.
There are no third-party frontend dependencies or analytics. `build_data.py`
rebuilds the interactive chart payload from the aggregate CSVs in `dist/data/`.
Original data identifiers remain unchanged; the interface labels the reference SFT.

The charts use the published development-benchmark tables. The main curves are
descriptive seed means, and their thin lines are seed trajectories, not confidence
bands. Bootstrap confidence intervals are shown only at the common 22.51% point.
Overlap controls match hire counts rather than transferring numeric thresholds.
The parameter-budget section retains the original served-checkpoint evaluation
basis; it does not substitute later HF merge diagnostics.

Keep `dist/poster.pdf` synchronized with the poster's compiled PDF. Update the
CSV snapshots and regenerate `results.js` together. The hosted Sites checkout keeps its identity in `.openai/hosting.json`.
This directory is a portable source snapshot; update the existing hosted site
when publishing changes, rather than creating a new site.

Scientific source repository:
https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at
