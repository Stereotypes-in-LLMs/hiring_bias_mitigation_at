"""CPU regression checks for comparing a newly scored mechanism checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location("distill_compare_followup",
                                              ROOT / "scripts/interp/20_distill_compare.py")
dc = importlib.util.module_from_spec(module)
module.loader.exec_module(dc)
guard_module = importlib.util.spec_from_file_location("mechanism_cache_guard",
                                                    ROOT / "scripts/check_mechanism_followup.py")
guard = importlib.util.module_from_spec(guard_module)
guard_module.loader.exec_module(guard)


def frames():
    manifest = pd.DataFrame({"contrast_id": ["x", "x"], "aligned": True,
                             "candidate_id": ["c1", "c2"],
                             "uid_a": ["a1", "a2"], "uid_b": ["b1", "b2"]})
    reference = pd.DataFrame({
        "prompt_uid": ["f1", "f2", "a1", "a2", "b1", "b2"],
        "candidate_id": ["c1", "c2", "c1", "c2", "c1", "c2"],
        "condition": ["attr_free", "attr_free", "implicit", "implicit", "implicit", "implicit"],
        "margin_fp32": [4.0, 8.0, 2.0, 4.0, 0.0, 0.0],
    })
    compared = reference.copy()
    compared["margin_fp32"] *= 0.5
    return manifest, reference, compared


class MechanismFollowupChecks(unittest.TestCase):
    def run_sensitivity(self, manifest, reference, compared):
        return dc.sensitivity(manifest, reference, compared, "new_seed", "sft_v3", ["x"],
                              n_boot=20, seed=0).iloc[0]

    def test_new_spec_uses_aligned_margins_without_mutating_manifest(self):
        manifest, reference, compared = frames()
        original = manifest.copy(deep=True)
        out = self.run_sensitivity(manifest, reference, compared.iloc[::-1])
        self.assertAlmostEqual(out["r_attr"], 0.5)
        self.assertAlmostEqual(out["rho"], 1.0)
        pd.testing.assert_frame_equal(manifest, original)

    def test_stale_effect_columns_cannot_override_current_margins(self):
        manifest, reference, compared = frames()
        manifest["dm_new_seed"], manifest["dm_sft_v3"] = 1000.0, 1.0
        self.assertAlmostEqual(self.run_sensitivity(manifest, reference, compared)["r_attr"], 0.5)

    def test_missing_or_duplicate_prompts_are_rejected(self):
        manifest, reference, compared = frames()
        for invalid in (compared.iloc[:-1], pd.concat([compared, compared.iloc[:1]])):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                self.run_sensitivity(manifest, reference, invalid)

    def test_cross_candidate_pair_is_rejected(self):
        manifest, reference, compared = frames()
        manifest.loc[0, "uid_b"] = "b2"
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            self.run_sensitivity(manifest, reference, compared)

    def test_missing_attribute_free_candidate_is_not_silently_dropped(self):
        manifest, reference, compared = frames()
        reference = reference[reference.prompt_uid != "f2"]
        compared = compared[compared.prompt_uid != "f2"]
        with self.assertRaisesRegex(ValueError, "attribute-free"):
            self.run_sensitivity(manifest, reference, compared)


class MechanismCacheChecks(unittest.TestCase):
    def test_new_partial_cache_reuses_only_the_bound_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            guard.cache_identity(path, {"adapter": "first"}, prepare=True)
            (path / "part-000.parquet").write_bytes(b"partial")
            guard.cache_identity(path, {"adapter": "first"})
            with self.assertRaisesRegex(ValueError, "identity changed"):
                guard.cache_identity(path, {"adapter": "other"})

    def test_ambiguous_partial_cache_is_refused_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            part = path / "part-000.parquet"
            part.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                guard.cache_identity(path, {"adapter": "current"}, prepare=True)
            self.assertEqual(part.read_bytes(), b"partial")
            self.assertFalse((path / "poster_followup.meta.json").exists())

    def test_legacy_reuse_is_explicit_and_still_detects_later_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "verify.parquet").write_bytes(b"old")
            record = guard.cache_identity(path, {"adapter": "current"},
                                          prepare=True, legacy_allowed=True)
            self.assertEqual(record["prior_cache_provenance"], "legacy_cache_adapter_identity_unverified")
            with self.assertRaises(ValueError):
                guard.cache_identity(path, {"adapter": "changed"}, legacy_allowed=True)

    def test_native_fingerprint_is_checked_even_when_summary_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            native = path / "grid-en.parquet.meta.json"
            native.write_text(json.dumps({"adapters": {"adapter": "old"}}))
            with self.assertRaisesRegex(ValueError, "native adapter fingerprint"):
                guard.cache_identity(path, {"adapter": "new"}, native_meta=native)


if __name__ == "__main__":
    unittest.main()
