"""Numerical safeguards for the poster's CPU reanalysis, without model dependencies."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("poster_followup", ROOT / "scripts/poster_audit_followup.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PosterFollowupChecks(unittest.TestCase):
    def test_ties_are_whole_sets_and_hire_count_is_exact(self):
        opc = audit.load_matching()
        frame = pd.DataFrame({
            "pair_id": ["c1__j", "c1__j", "c2__j", "c2__j"],
            "protected_group": "gender", "condition": "implicit",
            "protected_attr": ["A", "B", "A", "B"],
            "margin": [1.0, 0.0, 0.0, -1.0],
        })
        flags, cut = opc.matched_flags(frame, "margin", 2)
        self.assertEqual(cut["hire_rate"], 0.5)
        self.assertEqual(cut["tie_weight"], 0.5)
        np.testing.assert_allclose(flags, [0.5, 0.5])
        # A pair of identically tied prompts is stable at both neighbouring cuts.
        frame["margin"] = [0.0, 0.0, 1.0, -1.0]
        flags, _ = opc.matched_flags(frame, "margin", 2)
        self.assertEqual(flags.loc[("c1__j", "gender", "implicit")], 0.0)

    def test_weighted_overlap_is_not_binaryized(self):
        a = pd.Series([0.5, 1.0, 0.0], index=["a", "b", "c"])
        b = pd.Series([1.0, 0.5, 1.0], index=["a", "b", "c"])
        shared, union, jac = audit.weighted_jaccard(a, b)
        self.assertEqual(shared, 1.0)
        self.assertEqual(union, 3.0)
        self.assertAlmostEqual(jac, 1 / 3)
        self.assertEqual(audit.weighted_jaccard(a, a)[2], 1.0)

    def test_real_ranks_can_reverse_attribution_ranks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "reports/interp/patching/dpo4880-grid-mil_combat_civ_impl-n64-s42-sel_sft_v3"
            d.mkdir(parents=True)
            pd.DataFrame({"pair_uid": ["p1", "p2"]}).to_csv(d / "pairs.csv", index=False)
            atp = pd.DataFrame({"kind": "head_in", "cls": "suffix", "layer": [23, 15, 7, 6],
                                "head": [1, 11, 3, 1], "removed_logits_per_pair": [0.9, 0.5, 0.3, 0.1]})
            atp.to_csv(d / "atp.csv", index=False)
            real = pd.DataFrame({"cls": "suffix", "layer": [23, 15, 6], "head": [1, 11, 1],
                                 "control": [False, False, True],
                                 "removed_logits_per_pair": [0.1, 0.5, 0.02],
                                 "share_removed": [0.1, 0.5, 0.02]})
            real.to_csv(d / "verify.csv", index=False)
            out = root / "out"
            out.mkdir()
            audit.real_head_checks(root, out)
            target = pd.read_csv(out / "target_heads.csv").set_index("target")
            self.assertEqual(target.loc["L23.H1", "atp_rank_all"], 1)
            self.assertEqual(target.loc["L23.H1", "real_rank_tested"], 2)
            self.assertEqual(target.loc["L15.H11", "real_rank_tested"], 1)
            self.assertEqual(target.loc["L15.H11", "verified_heads"], 3)
            self.assertEqual(target.loc["L15.H11", "all_atp_heads"], 4)
            # A head absent from real patches must remain unverified, never a zero effect.
            real[real.layer != 23].to_csv(d / "verify.csv", index=False)
            audit.real_head_checks(root, out)
            target = pd.read_csv(out / "target_heads.csv").set_index("target")
            self.assertFalse(target.loc["L23.H1", "verified"])
            self.assertTrue(pd.isna(target.loc["L23.H1", "real_rank_tested"]))


if __name__ == "__main__":
    unittest.main()
