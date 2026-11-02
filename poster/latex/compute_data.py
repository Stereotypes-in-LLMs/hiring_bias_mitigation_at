"""Poster compute data, regenerated from measured result tables.

Pair-count metadata is required for 64-pair cost claims: ``--n-pairs`` is a cap,
not the selected sample size. DDP repeat selection is explicit in ddp_review.json.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import shlex

import pandas as pd


def tex_table(name: str, records: list[dict]) -> str:
    columns = list(records[0])
    if any("," in str(row[key]) for row in records for key in columns):
        raise ValueError(f"Comma in {name} table value")
    # These tables are parsed in the preamble. Keep the same short escape list
    # that the poster's existing scientific tables have been compiled with.
    for row in records:
        for key in columns:
            value = str(row[key])
            for allowed in ("\\textminus{}", "\\texttimes", "\\%"):
                value = value.replace(allowed, "")
            if "\\" in value:
                raise ValueError(f"Unapproved TeX command in {name}.{key}: {row[key]}")
    data = "\n".join(",".join(str(row[key]) for key in columns) for row in records)
    return f"\\pgfplotstableread[col sep=comma]{{\n{','.join(columns)}\n{data}\n}}\\{name}\n"


def macro_lines(findings: dict) -> list[str]:
    for name in findings:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"Invalid ordinary TeX macro name: {name}")
    return [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in findings.items()]


def ddp_phases(root: Path) -> dict:
    """One-sided late-slowdown diagnostics, without choosing a baseline.

    The user accepted this post-hoc interpretation; ddp_review.json records that
    it differs from the registration's original absolute 5% steady-state bound.
    """
    times = pd.read_csv(root / "reports/throughput_b/accel/step_times.csv")
    phases = {}
    for label in ("bench-distill-ddp1", "bench-distill-ddp1-r2", "bench-distill-ddp4"):
        rows = times[(times["bench"] == "B2") & (times["label"] == label)]
        checkpoints = rows[rows["step"].isin([5, 40])]
        if len(checkpoints) != 2 or checkpoints["step"].nunique() != 2:
            phases[label] = {"state": "pending", "reason": "missing complete step-5/40 timing"}
            continue
        checkpoints = checkpoints.set_index("step")["elapsed_s"]
        early, late = float(checkpoints.loc[5] / 5), float((checkpoints.loc[40] - checkpoints.loc[5]) / 35)
        if not (early > 0 and late > 0):
            raise ValueError(f"Invalid DDP phase timing for {label}")
        change = late / early - 1
        phases[label] = dict(state="no late slowdown" if change <= 0.05 else "late slowdown",
                             early_s=early, late_s=late, fractional_change=change)
    return phases


def patch_identity(command: str) -> dict:
    """Use the same defaults and run-name construction as scripts/interp/10_patch.py."""
    words = shlex.split(command)
    options = {}
    for i, word in enumerate(words):
        if word.startswith("--"):
            if "=" in word:
                key, value = word.split("=", 1)
            elif i + 1 < len(words) and not words[i + 1].startswith("--"):
                key, value = word, words[i + 1]
            else:
                continue
            options[key] = value
    required = ("--spec", "--dataset", "--contrast")
    if any(key not in options for key in required):
        raise ValueError(f"Cannot identify patching run from command: {command}")
    spec, dataset, contrast = (options[key] for key in required)
    requested = int(options.get("--n-pairs", 64))
    seed = int(options.get("--seed", 42))
    select_on = options.get("--select-on", "base")
    run = f"{contrast}-n{requested}-s{seed}"
    if select_on != "base":
        run += f"-sel_{select_on}"
    return dict(spec=spec, dataset=dataset, run=run, requested=requested,
                report=f"{spec}-{dataset}-{run}")


def patch_cost_cohort(steps: pd.DataFrame, root: Path, output_root: Path) -> pd.DataFrame:
    """Return successful measured patch runs, with verified actual pair counts.

    A requested count below 64 is sufficient to exclude a pilot without its pair
    file. Every potentially eligible run must have selected-pair metadata.
    """
    runs = steps[(steps["stage"] == "patching") & (steps["source"] == "record")
                 & (steps["status"] == "ok") & steps["label"].str.startswith("patch-")].copy()
    counts, sources, missing = [], [], []
    for _, row in runs.iterrows():
        identity = patch_identity(str(row["command"]))
        if identity["requested"] < 64:
            counts.append(None)
            sources.append(f"excluded: requested at most {identity['requested']} pairs")
            continue
        report = root / "reports/interp/patching" / identity["report"] / "pairs.csv"
        raw = (output_root / "outputs/interp/patching" / identity["spec"]
               / identity["dataset"] / identity["run"] / "selected_pairs.parquet")
        if report.exists():
            pairs = pd.read_csv(report, usecols=["pair_uid"])
            source = str(report.relative_to(root))
        elif raw.exists():
            pairs = pd.read_parquet(raw, columns=["pair_uid"])
            source = str(raw)
        else:
            missing.append(str(report))
            counts.append(None)
            sources.append("missing metadata")
            continue
        n = pairs["pair_uid"].nunique()
        if n != len(pairs) or not 0 < n <= identity["requested"]:
            raise ValueError(f"Invalid selected pair IDs/count in {source}")
        counts.append(n)
        sources.append(source)
    if missing:
        raise FileNotFoundError("Actual patch pair counts are missing; refusing a 64-pair claim:\n"
                                + "\n".join(missing))
    runs["actual_pairs"] = counts
    runs["pair_count_source"] = sources
    return runs


def compute_data(root: Path, output_root: Path | None = None) -> tuple[dict, dict]:
    """Return tables and macros; no result, source or configuration file is modified."""
    output_root = output_root or Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
    steps = pd.read_csv(root / "reports/cost/steps.csv", dtype={"job_id": str})
    stage = pd.read_csv(root / "reports/cost/by_stage.csv").set_index("stage")
    full = pd.read_csv(root / "reports/throughput_b/full/speedup_summary.csv")
    engine = full[(full["condition"] == "cacheon-asbuilt-nostats")
                  & (full["ref_condition"] == "cacheoff-shuffle")].iloc[0]
    findings = {
        "AuditWallS": f"{engine['wall_s_median']:.0f}",
        "AuditGenS": f"{engine['generate_s_median']:.0f}",
        "AuditReqPerS": f"{engine['requests_per_s_median']:.2f}",
        "AuditKWh": f"{engine['gpu_energy_kwh_median']:.3f}",
        "FullAuditPrompts": f"{int(engine['requests_median']):,}",
        "FullAuditRepeats": str(int(engine["repeats"])),
        "PrefixSpeedup": f"{engine['speedup_x_median']:.2f}",
        "ProjectGPUh": f"{stage.loc['total allocated', 'gpu_h']:.0f}",
        "ProjectKWh": f"{stage['gpu_energy_kwh'].sum():.0f}",
    }

    cohort = patch_cost_cohort(steps, root, output_root)
    sixty_four = cohort[cohort["actual_pairs"] == 64]
    if sixty_four.empty:
        raise ValueError("No verified 64-pair cost measurements")
    margins = steps[(steps["stage"] == "patching") & (steps["source"] == "record")
                    & (steps["status"] == "ok") & steps["label"].str.startswith("margins-")
                    & (steps["wall_s"] > 600)]
    if margins.empty:
        raise ValueError("No full-grid margin cost measurements")
    findings.update({
        "PatchRunsSixtyFour": str(len(sixty_four)),
        "PatchStepMin": f"{sixty_four['wall_s'].median() / 60:.0f}",
        "PatchStepGPUh": f"{sixty_four['gpu_h'].median():.2f}",
        "PatchPeakGiB": f"{sixty_four['mem_peak_gib'].max():.0f}",
        "PatchStepMinLow": f"{sixty_four['wall_s'].min() / 60:.1f}",
        "PatchStepMinHigh": f"{sixty_four['wall_s'].max() / 60:.1f}",
        "PatchStepsOK": str(len(cohort)),
        "MarginsStepMin": f"{margins['wall_s'].median() / 60:.0f}",
        "MarginsPeakGiB": f"{margins['mem_peak_gib'].max():.0f}",
        "PatchTotalGPUh": f"{stage.loc['patching', 'gpu_h']:.1f}",
    })

    speed = pd.read_csv(root / "reports/throughput_b/accel/speedup_summary.csv")
    audit_rows = []
    for key, label, against in (("hf8", "HF batch 8", "vllm"),
                                 ("hf32", "HF batch 32", "vllm"),
                                 ("vllm", "vLLM", "hf8")):
        record = speed[(speed["condition"] == key) & (speed["ref_condition"] == against)].iloc[0]
        audit_rows.append(dict(key=key, label=label, seconds=round(record["generate_s_median"], 3),
                               lo=round(record["generate_s_min"], 3),
                               hi=round(record["generate_s_max"], 3)))
    vllm = speed[(speed["condition"] == "vllm") & (speed["ref_condition"] == "hf8")].iloc[0]
    vllm32 = speed[(speed["condition"] == "vllm") & (speed["ref_condition"] == "hf32")].iloc[0]
    identities = pd.read_csv(root / "reports/throughput_b/accel/identity.csv")
    hf_a = identities["run_a"].str.contains(r"throughput-accel-hf(?:8|32)-")
    hf_b = identities["run_b"].str.contains(r"throughput-accel-hf(?:8|32)-")
    vl_a = identities["run_a"].str.contains("throughput-accel-vllm-")
    vl_b = identities["run_b"].str.contains("throughput-accel-vllm-")
    agreement = 100 * identities.loc[(hf_a & vl_b) | (vl_a & hf_b), "same_decision_share"]
    if agreement.empty:
        raise ValueError("Missing HF versus vLLM decision-identity comparisons")
    findings.update({
        "AuditBenchPrompts": f"{int(vllm['requests_median']):,}",
        "AuditBenchRepeats": str(int(vllm["repeats"])),
        "AuditSpeedSmall": f"{vllm['speedup_x_median']:.1f}",
        "AuditSpeedLarge": f"{vllm32['speedup_x_median']:.1f}",
        "AuditBackendAgree": f"{agreement.min():.2f}--{agreement.max():.2f}",
        "AuditHFSmallSeconds": f"{audit_rows[0]['seconds']:.0f}",
        "AuditHFLargeSeconds": f"{audit_rows[1]['seconds']:.0f}",
        "AuditVLLMSeconds": f"{audit_rows[2]['seconds']:.1f}",
    })

    review_path = root / "poster/latex/ddp_review.json"
    review = json.loads(review_path.read_text())
    if review.get("status") not in ("pending", "reviewed"):
        raise ValueError(f"Invalid DDP review status in {review_path}")
    records = steps[(steps["source"] == "record") & (steps["status"] == "ok")]

    def selected(key: str) -> pd.Series:
        selection = review[key]
        hit = records[(records["label"] == selection["label"])
                      & (records["job_id"] == str(selection["job_id"]))]
        if len(hit) != 1:
            raise ValueError(f"DDP {key} selection must identify one successful record: {selection}")
        return hit.iloc[0]

    one, four = selected("one_gpu"), selected("four_gpu")
    if one["gpu_basis"] != "used" or four["gpu_basis"] != "used":
        raise ValueError("DDP cost comparison requires used GPU-h for both records")
    if int(one["n_gpus"]) != 1 or int(four["n_gpus"]) != 4:
        raise ValueError("DDP selections must use one and four GPUs respectively")
    for key in ("train_steps", "effective_batch", "precision"):
        if one[key] != four[key]:
            raise ValueError(f"DDP workload mismatch for {key}: {one[key]} vs {four[key]}")
    note = review.get("note", "")
    if any(c in note for c in "{}\\%&#_$"):
        raise ValueError("DDP note must be plain text without TeX special characters")
    findings.update({
        "DDPTrainingSpeedup": f"{one['train_runtime_s'] / four['train_runtime_s']:.2f}",
        "DDPEndToEndSpeedup": f"{one['wall_s'] / four['wall_s']:.2f}",
        "DDPOneSeconds": f"{one['train_runtime_s']:.0f}",
        "DDPFourSeconds": f"{four['train_runtime_s']:.0f}",
        "DDPOneWallSeconds": f"{one['wall_s']:.0f}",
        "DDPFourWallSeconds": f"{four['wall_s']:.0f}",
        "DDPOneGPUh": f"{one['gpu_h']:.3f}",
        "DDPFourGPUh": f"{four['gpu_h']:.3f}",
        "DDPCostIncrease": f"{100 * (four['gpu_h'] / one['gpu_h'] - 1):.0f}",
        "DDPCostDelta": f"{100 * (four['gpu_h'] / one['gpu_h'] - 1):+.0f}",
        "DDPTrainingCostDelta": f"{100 * (four['n_gpus'] * four['train_runtime_s'] / (one['n_gpus'] * one['train_runtime_s']) - 1):+.0f}",
        "DDPTrainingOneGPUh": f"{one['n_gpus'] * one['train_runtime_s'] / 3600:.3f}",
        "DDPTrainingFourGPUh": f"{four['n_gpus'] * four['train_runtime_s'] / 3600:.3f}",
        "DDPReviewed": str(int(review["status"] == "reviewed")),
        "DDPStatus": note,
        "DDPBatch": str(int(one["effective_batch"])),
        "DDPSteps": str(int(one["train_steps"])),
    })
    phases = ddp_phases(root)
    repeat = phases["bench-distill-ddp1-r2"]
    findings["DDPRepeatState"] = repeat["state"]
    for label, prefix in (("bench-distill-ddp1", "DDPFirst"),
                          ("bench-distill-ddp1-r2", "DDPRepeat"),
                          ("bench-distill-ddp4", "DDPFour")):
        phase = phases[label]
        findings[f"{prefix}EarlyS"] = f"{phase['early_s']:.1f}" if "early_s" in phase else "pending"
        findings[f"{prefix}LateS"] = f"{phase['late_s']:.1f}" if "late_s" in phase else "pending"
    return {"AuditBenchData": audit_rows}, findings


def write_compute_summary(root: Path, tables: dict, findings: dict,
                          output_root: Path | None = None) -> Path:
    """Write the measurement scope and the exact selected records beside other follow-ups."""
    output_root = output_root or Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
    steps = pd.read_csv(root / "reports/cost/steps.csv", dtype={"job_id": str})
    review = json.loads((root / "poster/latex/ddp_review.json").read_text())
    cohort = patch_cost_cohort(steps, root, output_root)
    cohort = cohort[cohort["actual_pairs"] == 64]
    selected = {}
    for key in ("one_gpu", "four_gpu"):
        choice = review[key]
        row = steps[(steps["source"] == "record") & (steps["status"] == "ok")
                    & (steps["label"] == choice["label"]) & (steps["job_id"] == str(choice["job_id"]))].iloc[0]
        selected[key] = {k: (float(row[k]) if k in ("wall_s", "train_runtime_s", "gpu_h") else str(row[k]))
                         for k in ("job_id", "label", "wall_s", "train_runtime_s", "gpu_h", "gpu_basis")}
    sources = ("reports/cost/steps.csv", "reports/cost/by_stage.csv",
               "reports/throughput_b/full/speedup_summary.csv",
               "reports/throughput_b/accel/speedup_summary.csv",
               "reports/throughput_b/accel/identity.csv",
               "reports/throughput_b/accel/step_times.csv", "poster/latex/ddp_review.json")
    summary = dict(
        sources_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        ddp_review=review, ddp_selected_records=selected, ddp_phase_diagnostics=ddp_phases(root),
        patch_sixty_four=dict(runs=len(cohort), median_minutes=float(cohort["wall_s"].median()/60),
                              median_used_gpu_h=float(cohort["gpu_h"].median()),
                              peak_gib=float(cohort["mem_peak_gib"].max()),
                              job_ids=sorted(set(cohort["job_id"])),
                              pair_count_sources=sorted(set(cohort["pair_count_source"]))),
        timing_scope={"audit_benchmark": "generation only; startup excluded",
                      "full_audit_wall": "startup, generation and writing; downstream scoring excluded",
                      "full_audit_requests_per_s": "generation throughput, not end-to-end throughput",
                      "ddp_training_speedup": "train_runtime; startup excluded",
                      "ddp_end_to_end_speedup": "cost-step wall time; startup included",
                      "ddp_gpu_h": "used GPU-h; startup included",
                      "ddp_training_gpu_h": "GPU count times train_runtime; startup excluded",
                      "project_gpu_h": "allocated GPU-h"},
        tables=tables, macros=findings,
    )
    path = root / "reports/poster_followup/compute_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


LEGACY_COMPUTE_MACROS = {
    "PatchRuns64", "AuditDetail", "TrainDetail", "GRPODetail", "TrainOneEarly",
    "TrainOneLate", "TrainRatioEarly", "TrainRatioLate", "RolloutTokens", "RolloutSavedS",
    "RolloutStepS", "TrainExtraGPUh", "TrainState", "TrainNote",
}


def refresh_compute_block(text: str, tables: dict, findings: dict) -> str:
    """Replace only compute-owned data in the existing generated block."""
    begin, end = "% BEGIN GENERATED DATA\n", "% END GENERATED DATA\n"
    head, rest = text.split(begin, 1)
    block, tail = rest.split(end, 1)
    lines, kept, i = block.splitlines(keepends=True), [], 0
    while i < len(lines):
        if lines[i].startswith("\\pgfplotstableread[col sep=comma]{"):
            j = i + 1
            while j < len(lines) and not re.fullmatch(r"\}\\[A-Za-z]+\s*", lines[j]):
                j += 1
            if j == len(lines):
                raise ValueError("Unterminated generated TeX table")
            name = lines[j].strip()[2:]
            if name not in {"AccelData", *tables}:
                kept.extend(lines[i:j + 1])
            i = j + 1
            continue
        match = re.match(r"\\newcommand\{\\([^}]+)\}", lines[i])
        if not match or match[1] not in {*findings, *LEGACY_COMPUTE_MACROS}:
            kept.append(lines[i])
        i += 1
    generated = "\n".join(tex_table(name, recs) for name, recs in tables.items())
    generated += "\n" + "\n".join(macro_lines(findings))
    return head + begin + "".join(kept).rstrip() + "\n\n" + generated + "\n" + end + tail
