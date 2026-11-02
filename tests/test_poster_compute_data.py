"""Regression checks for cost cohort membership and explicit benchmark selection."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "poster/latex"))
import compute_data as C


class ComputeDataTests(unittest.TestCase):
    def test_actual_pairs_include_base_selected_gender_and_exclude_short_selections(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = []
            for spec, contrast, select_on, requested, actual in (
                ("base", "gen_male_female_impl", "base", 64, 64),
                ("sft_v3", "gen_male_female_impl", "base", 64, 64),
                ("sft_v3", "gen_male_female_impl", "sft_v3", 64, 45),
                ("distill43", "mil_combat_civ_impl", "sft_v3", 4, 4),
            ):
                cmd = (f"python scripts/interp/10_patch.py --spec {spec} --dataset grid "
                       f"--contrast {contrast} --select-on {select_on} --n-pairs {requested}")
                records.append(dict(stage="patching", source="record", status="ok",
                                    label=f"patch-{spec}-{contrast}", command=cmd))
                identity = C.patch_identity(cmd)
                if requested >= 64:
                    path = root / "reports/interp/patching" / identity["report"] / "pairs.csv"
                    path.parent.mkdir(parents=True)
                    pd.DataFrame({"pair_uid": range(actual)}).to_csv(path, index=False)
            result = C.patch_cost_cohort(pd.DataFrame(records), root, root)
            self.assertEqual(list(result["actual_pairs"].dropna()), [64, 64, 45])
            self.assertEqual(sum(result["actual_pairs"] == 64), 2)
            self.assertIn("at most 4", result.iloc[-1]["pair_count_source"])

    def test_missing_selected_pair_metadata_is_an_error(self):
        frame = pd.DataFrame([dict(stage="patching", source="record", status="ok",
                                  label="patch-base-anything", command=
                                  "python 10_patch.py --spec base --dataset grid --contrast anything")])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "refusing a 64-pair claim"):
                C.patch_cost_cohort(frame, Path(td), Path(td))

    def test_refresh_preserves_scientific_data_and_is_idempotent(self):
        before = (ROOT / "main.tex").read_text() if (ROOT / "main.tex").exists() else (
            ROOT / "poster/latex/main.tex").read_text()
        tables = {"AuditBenchData": [dict(label="HF batch 8", seconds=1091)]}
        fields = {"PatchRunsSixtyFour": "22", "PatchPeakGiB": "42"}
        after = C.refresh_compute_block(before, tables, fields)
        self.assertIn("}\\ModelData", after)
        self.assertIn("}\\SeedData", after)
        self.assertIn("\\newcommand{\\NPairs}{450}", after)
        self.assertNotIn("\\PatchRuns64", after)
        self.assertNotIn("}\\AccelData", after)
        self.assertEqual(after, C.refresh_compute_block(after, tables, fields))

    def test_reviewed_repeat_updates_both_runtime_and_cost(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in (
                "reports/cost/steps.csv", "reports/cost/by_stage.csv",
                "reports/throughput_b/full/speedup_summary.csv",
                "reports/throughput_b/accel/speedup_summary.csv",
                "reports/throughput_b/accel/identity.csv",
                "reports/throughput_b/accel/step_times.csv",
                "poster/latex/ddp_review.json",
            ):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, target)
            steps_path = root / "reports/cost/steps.csv"
            steps = pd.read_csv(steps_path, dtype={"job_id": str})
            one = steps[steps["label"] == "bench-distill-ddp1"].iloc[0].copy()
            one["label"] = "bench-distill-ddp1-r2"
            one["job_id"] = "test-repeat"
            one["train_runtime_s"], one["wall_s"], one["gpu_h"] = 2000, 2100, 0.6
            pd.concat([steps, pd.DataFrame([one])]).to_csv(steps_path, index=False)
            review_path = root / "poster/latex/ddp_review.json"
            review = json.loads(review_path.read_text())
            review.update(status="reviewed", note="Repeated baseline inspected")
            review["one_gpu"] = dict(label="bench-distill-ddp1-r2", job_id="test-repeat")
            review_path.write_text(json.dumps(review))
            sample = pd.DataFrame([dict(actual_pairs=64, wall_s=1200, gpu_h=1/3,
                                        mem_peak_gib=42)])
            with patch.object(C, "patch_cost_cohort", return_value=sample):
                tables, fields = C.compute_data(root, root)
            self.assertEqual(fields["DDPOneSeconds"], "2000")
            self.assertEqual(fields["DDPOneGPUh"], "0.600")
            self.assertEqual(fields["DDPTrainingSpeedup"], "3.64")
            self.assertEqual(fields["DDPCostIncrease"], "57")
            self.assertEqual(fields["DDPTrainingCostDelta"], "+10")
            self.assertEqual(fields["DDPReviewed"], "1")
            self.assertEqual(fields["AuditSpeedLarge"], "7.3")
            self.assertEqual(fields["AuditReqPerS"], "45.65")
            self.assertEqual([r["key"] for r in tables["AuditBenchData"]],
                             ["hf8", "hf32", "vllm"])

    def test_preamble_tables_and_macro_names_reject_unsafe_tokens(self):
        with self.assertRaisesRegex(ValueError, "Invalid ordinary TeX macro"):
            C.macro_lines({"PatchRuns64": "22"})
        with self.assertRaisesRegex(ValueError, "Unapproved TeX command"):
            C.tex_table("Example", [{"name": r"before \textrightarrow{} after"}])
        self.assertIn(r"\textminus{}", C.tex_table("Example", [{"value": r"\textminus{}0.2"}]))


if __name__ == "__main__":
    unittest.main()
