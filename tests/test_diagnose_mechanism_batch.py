import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_mechanism_batch import choose_pairs, original_full_batches, pair_metrics


class BatchDiagnosticTests(unittest.TestCase):
    def test_original_full_batches_use_stable_character_length_sort(self):
        uids = ["a", "b", "c", "d", "e"]
        texts = {"a": "12345", "b": "x", "c": "12345", "d": "123", "e": "12"}
        self.assertEqual(original_full_batches(uids, texts, 2, {"a"}), [(2, ["d", "a"])])
        self.assertEqual(original_full_batches(uids, texts, 2, {"c"}), [(4, ["c"])])

    def test_diagnostic_selection_has_disjoint_extremes(self):
        selected = pd.DataFrame({"pair_uid": [f"p{i}" for i in range(6)], "candidate_id": [f"c{i}" for i in range(6)],
                                 "uid_a": [f"a{i}" for i in range(6)], "uid_b": [f"b{i}" for i in range(6)]})
        pairs = selected[["pair_uid", "candidate_id"]].assign(m_a=list(range(6)), m_b=0.)
        margins = pd.DataFrame({"prompt_uid": selected.uid_a.tolist() + selected.uid_b.tolist(), "margin_fp32": [0.] * 12})
        chosen = choose_pairs(selected, pairs, margins, 4)
        self.assertEqual(chosen.pair_uid.tolist(), ["p5", "p4", "p0", "p1"])
        self.assertEqual(chosen.pair_label.tolist(), ["W01", "W02", "L01", "L02"])

    def test_crossing_and_ranks_keep_tie_at_zero_rejected(self):
        rows = [{"context": "singleton", "repeat": 1, "pair_label": label, "side": side, "margin": margin}
                for label, a, b in (("p1", 0., 1.), ("p2", -2., 1.))
                for side, margin in (("a", a), ("b", b))]
        result = pair_metrics(rows).set_index("pair_label")
        self.assertTrue(result.loc["p1", "crosses_zero"])
        self.assertEqual(result.loc["p1", "crossing_interval_low"], 0.)
        self.assertEqual(result.loc["p2", "abs_dm_rank_diagnostic_subset"], 1.)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            pair_metrics(rows[:-1])


if __name__ == "__main__":
    unittest.main()
