"""Plan or submit the registered poster follow-up jobs, with duplicate guards.

Defaults to a read-only plan. --submit validates CPU inputs, checks the live
queue, and records each accepted Slurm ID immediately. A recorded completed or
failed job is not silently submitted again; inspect its result before retrying.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager, nullcontext

ROOT = Path(__file__).resolve().parents[1]
SPECS = ("dpo4880_s43", "dpo4880_s44", "grpo_r4_s43", "grpo_r4_s44", "distill42", "distill44")
CONTRASTS = ("mil_combat_civ_impl", "mil_combat_civ_expl",
             "rel_atheist_christian_impl", "gen_male_female_impl")
JOB_NAMES = {f"xw-w163-{phase}{seed}" for phase in ("s", "e") for seed in (42, 43, 44)} | {
    f"xw-mech-{spec}" for spec in SPECS
}


def active_followups(queue_output: str) -> dict[str, str]:
    """Ignore other projects and arrays in the shared user's queue."""
    active = {}
    for row in queue_output.splitlines():
        job_id, name = row.strip().split("|", 1)
        if name not in JOB_NAMES:
            continue
        if name in active:
            raise SystemExit(f"multiple active jobs with name {name}; inspect queue first")
        active[name] = job_id
    return active


@contextmanager
def submission_lock():
    path = ROOT / "reports/experiment_followup/submissions.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another poster follow-up launcher is submitting; inspect it first") from exc
        yield


def adapter_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    with (directory / "adapter_model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("windows", "mechanisms", "all"), default="all")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    # The queue snapshot and ledger must be read under the same lock as submission.
    with submission_lock() if args.submit else nullcontext():
        run(args)


def run(args: argparse.Namespace) -> None:
    output_root = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
    ledger_path = ROOT / "reports/experiment_followup/submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"jobs": {}}
    active = {}
    if args.submit:
        queue = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%j"],
                               check=True, text=True, capture_output=True)
        active = active_followups(queue.stdout)
    environment = {**os.environ, "REPO": str(ROOT), "XW": str(output_root),
                   "HBM_OUTPUT_ROOT": str(output_root), "HBM_COST_ARM": "poster_followup"}

    def validate(command: list[str]) -> None:
        if args.submit:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, check=True)

    def enqueue(name: str, script: str, wall: str, extra: dict[str, str],
                dependency: str | None = None) -> str:
        if name in active:
            print(f"REUSE active {name}: {active[name]}")
            return active[name]
        if name in ledger["jobs"]:
            old = ledger["jobs"][name]
            raise SystemExit(f"{name} was already submitted as {old['job_id']}; inspect its result. "
                             "This launcher does not automatically retry or erase its ledger.")
        command = [str(ROOT / "scripts/cscs/submit.sh"), "--parsable", "--partition=normal",
                   f"--time={wall}", f"--job-name={name}"]
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        command.append(script)
        if not args.submit:
            print(json.dumps({"job": name, "wall_cap": wall, "environment": extra,
                              "command": command}, ensure_ascii=False))
            return f"<{name}>"
        accepted = subprocess.run(command, cwd=ROOT, env={**environment, **extra}, check=True,
                                  text=True, capture_output=True).stdout.strip()
        job_id = accepted.split(";")[0]
        if not job_id.isdigit():
            raise SystemExit(f"Unrecognised sbatch result {accepted!r}; inspect queue before retrying")
        ledger["jobs"][name] = {
            "job_id": job_id, "script": script, "wall_cap": wall,
            "environment": extra, "dependency": dependency,
            "submitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ledger_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(ledger_path)
        active[name] = job_id
        print(f"SUBMITTED {name}: {job_id}", flush=True)
        return job_id

    if args.group in ("windows", "all"):
        # Complete all input checks before submitting the first expensive job.
        for seed in (42, 43, 44):
            validate(["scripts/verify_window_budget.py", "--phase", "before", "--config",
                      f"configs/mitigation/distill/decision_distill_l15_20_r163_seed{seed}.yaml"])
        for seed in (42, 43, 44):
            directory = output_root / f"outputs/distill/decision_distill_l15_20_r163_seed{seed}"
            manifest = directory / "run_config.json"
            trained = (manifest.exists() and json.loads(manifest.read_text()).get("status") == "done"
                       and (directory / "adapter_model.safetensors").exists())
            training_id = None
            if trained:
                validate(["scripts/verify_window_budget.py", "--phase", "after", "--config",
                          f"configs/mitigation/distill/decision_distill_l15_20_r163_seed{seed}.yaml"])
                print(f"REUSE trained window-r163 seed {seed}")
            else:
                training_id = enqueue(f"xw-w163-s{seed}", "scripts/cscs/train_window_budget.sbatch",
                                      "02:00:00", {"POSTER_SEED": str(seed)})
            completed = ROOT / f"reports/experiment_followup/window_evaluation_seed{seed}.json"
            if completed.exists() and json.loads(completed.read_text()).get("completed") and trained:
                marker = json.loads(completed.read_text())
                if args.submit and marker.get("adapter_fingerprint") != adapter_fingerprint(directory):
                    raise SystemExit(f"evaluation marker has missing or changed adapter fingerprint: {completed}")
                print(f"REUSE evaluated window-r163 seed {seed}")
            else:
                enqueue(f"xw-w163-e{seed}", "scripts/cscs/evaluate_window_budget.sbatch",
                        "01:00:00", {"POSTER_SEED": str(seed)}, training_id)

    if args.group in ("mechanisms", "all"):
        for spec in SPECS:
            validate(["scripts/check_mechanism_followup.py", "--spec", spec, "--check", "inputs"])
        for spec in SPECS:
            complete = ROOT / f"reports/experiment_followup/mechanism_{spec}.json"
            if complete.exists() and json.loads(complete.read_text()).get("completed"):
                validate(["scripts/check_mechanism_followup.py", "--spec", spec, "--check", "margins"])
                for contrast in CONTRASTS:
                    validate(["scripts/check_mechanism_followup.py", "--spec", spec,
                              "--check", "patch", "--contrast", contrast])
                print(f"REUSE completed mechanism follow-up: {spec}")
                continue
            enqueue(f"xw-mech-{spec}", "scripts/cscs/poster_mechanism.sbatch",
                    "01:30:00", {"POSTER_SPEC": spec})


if __name__ == "__main__":
    main()
