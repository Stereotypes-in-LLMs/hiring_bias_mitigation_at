"""Submission must exclude a second launcher before it can inspect the queue."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def load_launcher():
    script = Path(__file__).resolve().parents[1] / "scripts/submit_poster_followups.py"
    spec = importlib.util.spec_from_file_location("poster_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script


def test_concurrent_launcher_is_rejected_and_lock_releases(tmp_path):
    module, script = load_launcher()
    module.ROOT = tmp_path
    child = """
import runpy, sys
from pathlib import Path
lock = runpy.run_path(sys.argv[1])["submission_lock"]
lock.__wrapped__.__globals__["ROOT"] = Path(sys.argv[2])
with lock():
    print("acquired")
"""
    command = [sys.executable, "-c", child, str(script), str(tmp_path)]
    with module.submission_lock():
        second = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert second.returncode != 0
        assert "another poster follow-up launcher" in second.stderr
        assert "acquired" not in second.stdout
    after_release = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert after_release.returncode == 0, after_release.stderr
    assert after_release.stdout.strip() == "acquired"


def test_unrelated_same_named_array_jobs_do_not_block_planned_followups():
    module, _ = load_launcher()
    unrelated = "\n".join(("100_1|kagomeR63seg2", "100_2|kagomeR63seg2",
                            "101_1|xw-mech-unrelated", "101_2|xw-mech-unrelated"))
    expected = {name: str(200 + i) for i, name in enumerate(sorted(module.JOB_NAMES))}
    queue = unrelated + "\n" + "\n".join(f"{job}|{name}" for name, job in expected.items())
    assert len(expected) == 12
    assert module.active_followups(queue) == expected


def test_duplicate_planned_job_names_still_refuse_submission():
    module, _ = load_launcher()
    for name in module.JOB_NAMES:
        with pytest.raises(SystemExit, match=f"multiple active jobs with name {name}"):
            module.active_followups(f"100_1|kagomeR63seg2\n100_2|kagomeR63seg2\n200|{name}\n201|{name}")
