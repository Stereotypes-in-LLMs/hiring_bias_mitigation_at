import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize_adapter_merge_effect as S


def test_fp32_threshold_does_not_acquire_bf16_rounding():
    opc = S.P.load_matching()
    frame = pd.DataFrame({"pair_id": ["a__j", "a__j", "b__j", "b__j"],
                          "protected_group": ["g"] * 4, "condition": ["implicit"] * 4,
                          "margin_fp32": [0.0001, 0.0002, 0.0003, 0.0004]})
    flags, cut = S.flags_at_count(frame, "margin_fp32", 1, opc)
    assert cut["threshold"] == 0.0003
    assert flags.tolist() == [0., 1.]


def test_tied_identical_identity_scores_never_create_instability():
    opc = S.P.load_matching()
    frame = pd.DataFrame({"pair_id": ["a__j", "a__j", "b__j", "b__j"],
                          "protected_group": ["g"] * 4, "condition": ["implicit"] * 4,
                          "margin_bf16": [0., 0., 1., 1.]})
    flags, cut = S.flags_at_count(frame, "margin_bf16", 1, opc)
    assert cut["tie_weight"] == .5
    assert flags.sum() == 0


def test_gap_change_cancels_a_shared_representation_shift():
    idx = pd.MultiIndex.from_tuples([("a__j", "g", "implicit"), ("b__j", "g", "implicit")],
                                    names=["pair_id", "protected_group", "condition"])
    full = pd.Series([0., .2], index=idx)
    window = pd.Series([.5, .3], index=idx)
    result = S.difference_in_differences(window + .1, full + .1, window, full)
    assert np.allclose(result, 0)
    result = S.difference_in_differences(window - .1, full, window, full)
    assert np.allclose(result, -.1)


def test_alignment_rejects_same_size_wrong_cases():
    a = pd.DataFrame({"key": ["a", "b"]})
    with pytest.raises(ValueError, match="align"):
        S.check_alignment(a, pd.DataFrame({"key": ["a", "c"]}), ["key"])


def test_alignment_accepts_equal_strings_with_different_storage_dtypes():
    a = pd.DataFrame({"key": pd.Series(["a", "b"], dtype="object")})
    b = pd.DataFrame({"key": pd.Series(["a", "b"], dtype="string")})
    S.check_alignment(a, b, ["key"])
