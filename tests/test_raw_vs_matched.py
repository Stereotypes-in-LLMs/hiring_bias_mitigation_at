"""Raw-versus-matched table: greedy set flags and the paired candidate bootstrap."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "raw_vs_matched.py"
_spec = importlib.util.spec_from_file_location("raw_vs_matched", _path)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _frame(decisions: dict) -> pd.DataFrame:
    rows = []
    for (pair, group, cond), values in decisions.items():
        for i, d in enumerate(values):
            rows.append({"pair_id": pair, "protected_group": group, "condition": cond,
                         "protected_attr": f"a{i}", "decision": d})
    rows.append({"pair_id": "c1__j1", "protected_group": "", "condition": "attr_free",
                 "protected_attr": "", "decision": "reject"})
    return pd.DataFrame(rows)


def test_a_set_is_unstable_when_its_variants_disagree_and_undecided_sets_drop():
    frame = _frame({("c1__j1", "gender", "explicit"): ["hire", "reject"],
                    ("c1__j1", "gender", "implicit"): ["hire", "hire"],
                    ("c2__j1", "gender", "explicit"): ["Hire", "invalid output"]})
    flags = R.greedy_flags(frame)
    assert flags.loc[("c1__j1", "gender", "explicit")] == 1.0
    assert flags.loc[("c1__j1", "gender", "implicit")] == 0.0
    assert np.isnan(flags.loc[("c2__j1", "gender", "explicit")])
    assert len(flags) == 3  # the attribute-free row forms no set


def test_paired_delta_drops_sets_undecided_in_either_run_and_clusters_by_candidate():
    idx = pd.MultiIndex.from_tuples(
        [("c1__j1", "g", "explicit"), ("c1__j2", "g", "explicit"), ("c2__j1", "g", "explicit"),
         ("c3__j1", "g", "explicit")], names=R.SET)
    run = pd.Series([1.0, 0.0, 0.0, np.nan], index=idx)
    ref = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    out = R.paired_delta(run, ref, n_boot=200)
    assert out["sets"] == 3
    assert out["delta_pp"] == pytest.approx(100 * (1 / 3 - 1))
    assert out["ci_low"] <= out["delta_pp"] <= out["ci_high"]
