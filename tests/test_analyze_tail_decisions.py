"""Synthetic boundary cases for the CPU-only research follow-up."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_tail_decisions as A


class DecisionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opc = A.P.load_matching()

    def example(self):
        # In one pair, two sets alternate instability as all tied prompts change
        # together. Pair/CV any must remain 1, not max(.3, .7) = .7.
        return pd.DataFrame({
            "pair_id": ["c1__j1"] * 4 + ["c2__j2"] * 2,
            "protected_group": ["g1", "g1", "g2", "g2", "g1", "g1"],
            "protected_attr": ["a", "b"] * 3,
            "condition": ["implicit"] * 6,
            "margin": [-1., 0., 0., 1., -2., -1.],
            "m": [-1., 0., 0., 1., -2., -1.],
        })

    def test_union_before_tie_interpolation(self):
        flags = A.memberships(self.example(), {"threshold": 0., "tie_weight": .3}, self.opc)
        self.assertAlmostEqual(flags["set"].iloc[0], .3)
        self.assertAlmostEqual(flags["set"].iloc[1], .7)
        self.assertEqual(flags["pair"].loc["c1__j1"], 1.)
        self.assertEqual(flags["cv"].loc["c1"], 1.)
        self.assertEqual(flags["cv"].loc["c2"], 0.)

    def test_canonical_set_matching_and_exact_count(self):
        frame = self.example()
        for count in range(len(frame) + 1):
            cut = A.calibration(frame.m, count, self.opc)
            got = A.memberships(frame, cut, self.opc)["set"]
            want, meta = self.opc.matched_flags(frame, "margin", count)
            np.testing.assert_allclose(got, want)
            self.assertAlmostEqual(meta["hire_rate"] * len(frame), count)

    def test_crossing_tie_is_global_not_independent_coin_flips(self):
        cut = {"threshold": 0., "tie_weight": .25}
        np.testing.assert_allclose(A.crossing([0, -1, 0], [0, 0, 1], cut), [0, .25, .75])

    def test_monotone_transform_preserves_rank_and_crossings(self):
        m = pd.Series([-3., -1., 0., 0., 2., 4.])
        transformed = m ** 3 + 17
        np.testing.assert_allclose(m.rank(method="average"), transformed.rank(method="average"))
        for count in range(len(m) + 1):
            a = A.calibration(m, count, self.opc)
            b = A.calibration(transformed, count, self.opc)
            np.testing.assert_allclose(A.crossing(m.iloc[:3], m.iloc[3:], a),
                                       A.crossing(transformed.iloc[:3], transformed.iloc[3:], b))

    def test_rejects_duplicate_or_misaligned_keys(self):
        frame = self.example()
        ref = A.validate_frame(frame, self.opc.KEY)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            A.validate_frame(pd.concat([frame, frame.iloc[:1]]), self.opc.KEY)
        changed = frame.copy()
        changed.loc[0, "protected_attr"] = "different"
        with self.assertRaisesRegex(ValueError, "differ"):
            A.validate_frame(changed, self.opc.KEY, ref)

    def test_identifier_validation_accepts_string_storage_dtypes(self):
        frame = self.example()
        frame["candidate_id"] = pd.Series(["c1"] * 4 + ["c2"] * 2, dtype="string")
        frame["job_id"] = pd.Series(["j1"] * 4 + ["j2"] * 2, dtype="string")
        A.validate_frame(frame, self.opc.KEY)
        frame.loc[0, "candidate_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            A.validate_frame(frame, self.opc.KEY)

    def test_leave_out_is_fixed_cut_exact_arithmetic(self):
        idx = pd.MultiIndex.from_tuples([("a__x", "g", "i"), ("a__y", "g", "i"),
                                         ("b__x", "g", "i")], names=self.opc.SET)
        delta = pd.Series([1., -.5, 0.], index=idx)
        cv = A.leave_one_out(delta, "cv").set_index("cluster_hash")
        self.assertEqual(cv.loc[A.opaque("cv", "a"), "delta_pp_after_removal"], 0.)
        self.assertEqual(cv.loc[A.opaque("cv", "b"), "delta_pp_after_removal"], 25.)
        job = A.leave_one_out(delta, "job").set_index("cluster_hash")
        self.assertEqual(job.loc[A.opaque("job", "x"), "delta_pp_after_removal"], -50.)

    def test_fuzzy_overlap_is_not_independent_probability(self):
        _, _, jac = A.P.weighted_jaccard(pd.Series([.5, 0.]), pd.Series([.5, 1.]))
        self.assertAlmostEqual(jac, 1 / 3)


if __name__ == "__main__":
    unittest.main()
