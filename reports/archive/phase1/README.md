# Phase-1 planning outputs, archived

These are the stage-1 analysis and the phase-1 mitigation plan of the audit study by Nazarii
Drushchak, generated on 2026-09-19 and 2026-09-27 (upstream `be9b4ce`), and four adapter
diagnostics from the same period. They were moved here on 2026-10-20 because no current code
reads them. Later arms are planned in the pre-registration of their own `docs/<EXPERIMENT>.md`
and judged by [`docs/EVAL_STANDARD.md`](../../../docs/EVAL_STANDARD.md).

| File | Written by | What it records |
|---|---|---|
| `ANALYSIS.md`, `analysis.json`, `mitigation_plan.yaml`, `enable_planned.sh` | `scripts/analyze_results.py` (`make local-analyze`) | which baseline cells carry evidence of bias, the targets and controls chosen for mitigation, and the selection script that enabled them |
| `EXPERIMENT_PLAN.md`, `experiment_runs.csv` | `scripts/make_experiment_matrix.py` (`make local-plan`) | every phase-1 run with its family, scope and config path |
| `adapter_diagnostic--*.json` | `scripts/diagnose_adapter.py` | HuggingFace + PEFT against vLLM on the same adapters, the evidence behind the serving-stack lesson (`reports/PAPER_KIT.md`, X2) |

The files are kept exactly as they were generated. Relative links inside them still point to
their old place: `RESULTS.md` there means `reports/RESULTS.md`.

Running `make local-analyze` or `make local-plan` again writes fresh versions to `reports/`,
computed over every scored run; these stay as the phase-1 record. `scripts/make_experiment_matrix.py`,
`scripts/select_phase2.py` and `scripts/run_mitigation_phase1.sh` read `reports/analysis.json`, so
they need `make local-analyze` first.
