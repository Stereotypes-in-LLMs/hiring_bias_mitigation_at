"""The declared full-arm operating point and delta sign must survive reporting."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import window_budget_results as results


class WindowBudgetResultsChecks(unittest.TestCase):
    def setUp(self):
        self.opc = results.load_matching()
        self.full = pd.DataFrame({"pair_id": ["c1__j", "c1__j", "c2__j", "c2__j"],
                                  "protected_group": "gender", "condition": "implicit",
                                  "protected_attr": ["a", "b", "a", "b"],
                                  "margin": [3.0, 2.0, -1.0, -2.0]})
        # At its own threshold window hires 3/4; at full's rate it hires 2/4
        # split across both sets, whereas full's selected set is stable.
        self.window = self.full.assign(margin=[3.0, -1.0, 2.0, 1.0])

    def test_primary_uses_full_rate_and_positive_means_window_worse(self):
        out = results.full_rate_comparison(self.full, self.window, self.opc, 50, 42)
        self.assertEqual(out["hire_rate_pct"], 50.0)
        self.assertEqual(out["delta_window_minus_full_pp"], 100.0)
        self.assertEqual(out["ci_low"], 100.0)
        self.assertEqual(out["ci_high"], 100.0)
        wrong_direction = self.opc.matched_comparison(self.full, self.window, n_boot=50)
        self.assertEqual(wrong_direction["hire_rate_run"], 0.75)

    def test_interval_sign_reversal_preserves_endpoint_order(self):
        class CanonicalStub:
            KEY = self.opc.KEY
            @staticmethod
            def matched_comparison(ref, run, **kwargs):
                return {"hire_rate_run": 0.5, "unstable_run": 0.1, "unstable_ref_matched": 0.2,
                        "delta_matched_pp": -10.0, "ci_low": -15.0, "ci_high": -2.0,
                        "ref_threshold_matched": 1.0, "ref_tie_weight": 0.5,
                        "prompts": 4, "sets": 2}
        out = results.full_rate_comparison(self.full, self.window, CanonicalStub(), 50, 42)
        self.assertEqual((out["ci_low"], out["ci_high"]), (2.0, 15.0))

    def test_common_rate_is_shared_and_row_order_does_not_matter(self):
        out = results.common_rate_comparison(self.full, self.window.iloc[::-1], 2,
                                             self.opc, 50, 42)
        self.assertEqual(out["hire_rate_pct"], 50.0)
        self.assertEqual(out["delta_arm_minus_full_pp"], 100.0)
        same = results.common_rate_comparison(self.full, self.full, 2, self.opc, 50, 42)
        self.assertEqual(same["delta_arm_minus_full_pp"], 0.0)

    def test_incomplete_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            results.full_rate_comparison(self.full, self.window.iloc[:-1], self.opc, 50, 42)


if __name__ == "__main__":
    unittest.main()
