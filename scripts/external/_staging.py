"""Shared pieces of the per-source staging scripts in this directory.

Each `stage_<slug>.py` turns one source's raw download into the staged tables that
`hiring_bias_mitigation.data.external` reads (cvs/jobs/pairs, plus any source-specific side
tables). They are ports of the scripts the acquiring agents ran on 2026-10-19 (kept outside git
under `$HBM_EXTERNAL_ROOT/_scripts/<slug>/`), with the fixes the verifiers later applied to the
staged files in place folded into the logic. A fresh fetch + stage therefore reproduces the
parquet that was verified, not the one the acquirer first wrote.

Rules every stager follows, because the raw files are untrusted downloads:

* raw files are parsed as data and never imported, executed or rendered. The one literal
  parser used is `ast.literal_eval`, for a CSV column that holds Python dict literals;
* a staged `text` is the raw string with only leading and trailing whitespace stripped;
* staging is deterministic: the same raw bytes give the same rows, order and values.

The stagers run under `data-tools-venv` (see `scripts/fetch_external_data.py`), because some
need rapidfuzz, openpyxl, datasketch or langdetect. None of those is a project dependency, and
nothing under `src/` imports these scripts: the project only reads the parquet they write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import pandas as pd

#: The interpreter that has the staging-only packages. The default is where it lives on the
#: CSCS machine the sources were acquired on; set HBM_DATA_TOOLS_PYTHON elsewhere.
DATA_TOOLS_PYTHON = os.environ.get(
    "HBM_DATA_TOOLS_PYTHON", "/ritom/scratch/cscs/zhang/xw/data-tools-venv/bin/python")

#: Snapshot of `generation/pool.py::_MARKERS["en"]` as it was when the sources were staged.
#: It is copied rather than imported so that a later change to the pool filter cannot silently
#: change what a re-stage writes into `extra`; tests/test_fetch_external.py fails when the two
#: drift apart, which is the moment to decide whether the staged extras should follow.
MARKERS_EN = [
    r"(?i)\bmarital status\b", r"(?i)\b(?:married|unmarried|divorced|widow(?:er|ed)?)\b",
    r"(?i)\bmilitary\b", r"(?i)\b(?:veteran|reservist|conscript|army|armed forces)\b",
    r"(?i)\breligio(?:n|us)\b",
    r"(?i)\b(?:christian|muslim|atheist|hindu|jew(?:ish)?|sikh|jain|buddhist|zoroastrian)\b",
    r"(?i)\bI am \d{1,2} years old\b", r"(?i)\b(?:he/him|she/her|they/them)\b",
    r"(?i)\bI identify as\b", r"(?i)\bmy pronouns\b",
    r"(?i)\b(?:female|male)\b(?!\s*(?:voice|model))",
]
MARKERS_EN_RE = [re.compile(p) for p in MARKERS_EN]


def marker_patterns(text: str) -> list[str]:
    """The marker patterns (as strings) that match `text`, in list order."""
    return [p for p, rx in zip(MARKERS_EN, MARKERS_EN_RE) if rx.search(text)]


def marker_indices(text: str) -> list[int]:
    """The positions in `MARKERS_EN` of the patterns that match `text`."""
    return [i for i, rx in enumerate(MARKERS_EN_RE) if rx.search(text)]


def require(*modules: str) -> None:
    """Fails early, naming the interpreter that has the module, instead of mid-way through."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise ModuleNotFoundError(
            f"staging needs {', '.join(missing)}, which the project venv does not have. "
            f"Run it with {DATA_TOOLS_PYTHON} -I instead."
        )


def write_table(df: pd.DataFrame, path: Path, compression: str = "snappy") -> None:
    """Writes one staged table atomically, so an interrupted stage never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def dumps(obj) -> str:
    """The JSON encoding every `extra` column uses: non-ASCII kept, key order as built."""
    return json.dumps(obj, ensure_ascii=False)


def main(stage, description: str, options: tuple[tuple[str, str], ...] = ()) -> None:
    """The command line every stager shares: `stage_<slug>.py <raw dir> <staged dir>`.

    `options` adds `--name PATH` flags for stagers that read a second source (FAIRE reads
    the Kaggle resume CSV); each is passed to `stage` as a keyword argument.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("raw", type=Path, help="the source's raw/ directory")
    parser.add_argument("out", type=Path, help="directory to write the staged parquet into")
    for flag, help_text in options:
        parser.add_argument(flag, type=Path, default=None, help=help_text)
    args = vars(parser.parse_args())
    raw, out = args.pop("raw"), args.pop("out")
    counts = stage(raw, out, **{k: v for k, v in args.items() if v is not None})
    print(json.dumps(counts, indent=1))
