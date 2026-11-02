"""Small regression checks for the scientific scopes used by the CPU summary."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_mechanism_seeds.py"
SPEC = importlib.util.spec_from_file_location("mechanism_seeds", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MechanismSeedSummaryTests(unittest.TestCase):
    def test_selection_checks_identity_not_only_pair_uid(self):
        expected = pd.DataFrame({"pair_uid": ["p1", "p2"], "candidate_id": ["c1", "c2"],
                                 "uid_a": ["a1", "a2"], "uid_b": ["b1", "b2"]})
        M.same_identity(expected, expected.iloc[::-1])
        swapped = expected.copy()
        swapped["uid_a"] = ["a2", "a1"]
        with self.assertRaisesRegex(ValueError, "identities differ"):
            M.same_identity(expected, swapped)

    def test_real_rank_is_signed_tested_only_and_unverified_is_missing(self):
        real = pd.DataFrame({"layer": [15, 0], "head": [11, 0], "cls": ["suffix"] * 2,
            "control": [False, True], "removed_logits_per_pair": [.2, -.8],
            "share_removed": [.4, -1.6], "ci_low": [.3, -1.8], "ci_high": [.5, -1.4]})
        atp = pd.DataFrame({"kind": ["head_in"] * 3, "layer": [15, 0, 23], "head": [11, 0, 1],
                           "cls": ["suffix"] * 3, "removed_logits_per_pair": [.1, .05, .9]})
        targets, overview = M.head_tables(real, atp)
        self.assertEqual(targets[0]["real_rank_tested"], 1)
        self.assertEqual(targets[0]["atp_rank_all"], 2)
        self.assertEqual(targets[0]["real_share_ci_low"], .3)
        self.assertFalse(targets[1]["verified"])
        self.assertIsNone(targets[1]["real_rank_tested"])
        self.assertEqual(overview["tested_heads"], 2)
        self.assertEqual(overview["control_heads"], 1)

    def test_effect_summary_keeps_negative_baseline_sign_and_rejects_missing_pairs(self):
        pairs = pd.DataFrame({"pair_uid": ["p1", "p2"], "m_a": [2., -2.], "m_b": [0., 0.]})
        effects = pd.DataFrame({"pair_uid": ["p1", "p2"], "layer": [15, 15], "m_patched": [1., -1.]})
        summary = M.effect_summary(pairs, effects, ["layer"])
        self.assertAlmostEqual(summary.removed_logits_per_pair.iat[0], 1.)
        self.assertAlmostEqual(summary.share_removed.iat[0], .5)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.effect_summary(pairs, effects.iloc[:1], ["layer"])

    def test_profile_is_pearson_of_matching_real_components(self):
        reference = pd.DataFrame({"kind": ["mixer"] * 32 + ["mlp"] * 32,
                                  "layer": list(range(32)) * 2, "cls": ["suffix"] * 64,
                                  "removed_logits_per_pair": np.arange(64, dtype=float)})
        arm = reference.copy()
        arm["removed_logits_per_pair"] = 3 * arm.removed_logits_per_pair + 2
        self.assertAlmostEqual(M.component_profile(reference, arm.iloc[::-1])["real_component_pearson_r"], 1.)
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            M.component_profile(reference, arm.iloc[:63])

    def test_baseline_discrepancy_is_measured_without_replacing_patch_values(self):
        selected = pd.DataFrame({"pair_uid": ["p1", "p2"], "candidate_id": ["c1", "c2"],
                                 "uid_a": ["a1", "a2"], "uid_b": ["b1", "b2"]})
        pairs = pd.DataFrame({"pair_uid": ["p1", "p2"], "candidate_id": ["c1", "c2"],
                              "m_a": [2., -1.], "m_b": [0., -2.]})
        margins = pd.DataFrame({"prompt_uid": ["a1", "b1", "a2", "b2"], "margin_fp32": [-1., 0., 1., -2.]})
        result = M.baseline_diagnostics(selected, pairs, margins)
        self.assertEqual(result["max_patch_vs_full_margin_abs_difference"], 3.)
        self.assertEqual(result["patch_vs_full_decision_disagreements"], 2)
        self.assertEqual(result["patch_vs_full_dm_sign_disagreements"], 1)
        self.assertEqual(result["selected_mean_abs_dm_patch"], 1.5)
        self.assertEqual(result["selected_mean_abs_dm_full"], 2.)

    def test_full_grid_tails_use_current_margins_not_stale_manifest_effects(self):
        manifest, ref, arm = [], [], []
        for contrast in M.CONTRASTS:
            for i in range(450):
                a, b, c = f"{contrast}/a{i}", f"{contrast}/b{i}", f"c{i // 3}"
                manifest.append({"contrast_id": contrast, "aligned": True, "candidate_id": c,
                                 "uid_a": a, "uid_b": b, "dm_sft_v3": 999999.})
                for uid, old, new in ((a, 2., 1. if i < 440 else 4.), (b, 0., 0.)):
                    common = {"prompt_uid": uid, "candidate_id": c, "condition": "implicit"}
                    ref.append({**common, "margin_fp32": old})
                    arm.append({**common, "margin_fp32": new})
        for i in range(150):
            common = {"prompt_uid": f"free{i}", "candidate_id": f"c{i}", "condition": "attr_free"}
            ref.append({**common, "margin_fp32": 2.})
            arm.append({**common, "margin_fp32": 4.})
        manifest, ref, arm = map(pd.DataFrame, (manifest, ref, arm))
        rows = M.full_grid_effects(manifest, ref, arm.iloc[::-1])
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["p50_ratio_vs_sft"], .5)
            self.assertEqual(row["p99_ratio_vs_sft"], 2.)
            self.assertEqual(row["r_scale"], 2.)
        with self.assertRaisesRegex(ValueError, "duplicate margin"):
            M.full_grid_effects(manifest, ref, pd.concat([arm, arm.iloc[:1]]))
        wrong = arm.copy()
        wrong.loc[0, "candidate_id"] = "different"
        with self.assertRaisesRegex(ValueError, "candidate_id mismatch"):
            M.full_grid_effects(manifest, ref, wrong)


if __name__ == "__main__":
    unittest.main()
