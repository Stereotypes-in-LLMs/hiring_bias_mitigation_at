# Hiring audit results website

Static research companion for the poster. Site URL:
https://xinyangwuethz.github.io/hiring-audit-results

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

Keep `dist/poster.pdf` synchronized with `../latex/main.pdf` (compact display
layout), and `dist/poster-print.pdf` with `../latex/main-print.pdf` (1-inch print
margins). Both versions are 43 × 24 inches. Update the
CSV snapshots and regenerate `results.js` together. GitHub Pages publishes the main branch root of
https://github.com/XinyangWuEthz/hiring-audit-results.
This directory is the research repository source snapshot. Copy dist contents
to the website repository root when publishing changes.

Scientific source repository:
https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at
