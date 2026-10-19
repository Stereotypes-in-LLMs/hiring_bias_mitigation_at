"""Compares one student's attribute-tipped mistakes between two on-policy runs on the same pairs.

    python scripts/compare_onpolicy_rows.py \\
        --base artifacts/semisynthetic-v2/onpolicy-it2 \\
        --new artifacts/semisynthetic-v2/onpolicy-gender-pi1 \\
        --group gender --out reports/gender_coverage

Both runs must hold rows.parquet, prompts.parquet, sample_report.json and raw/samples/ as
written by generate_onpolicy_full.py. Every row is re-scored from its raw samples: the share of
the k attributed samples that take the wrong decision, its pair's attribute-free right-rate, and
whether it is attribute-tipped with a wrong sample (onpolicy.select_negatives' rule, before the
covert filters). The base run is restricted to `--group`; only pairs present in both runs count.

Written to --out: summary.json (rates per run and their paired difference, overall, by
condition and by reference decision, with a pair bootstrap), by_attribute.csv, pairs.csv
(error rows per pair in each run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402


def row_stats(run: Path, chunk_size: int, lang: str = "en", min_free_right: float = 0.5,
              ) -> pd.DataFrame:
    """One line per attributed row: wrong samples of k, attr-free right-rate, tipped_wrong."""
    rows = pd.read_parquet(run / "rows.parquet")
    prompts = pd.read_parquet(run / "prompts.parquet")
    k = int(json.loads((run / "sample_report.json").read_text())["samples_per_prompt"])
    samples = OP.read_samples(run / "raw" / "samples", prompts["prompt"].tolist(), chunk_size, k)
    is_free = (prompts["kind"] == "attr_free").to_numpy()
    reference = rows.set_index("pair_id")["reference_decision"].groupby(level=0).first()
    free_right = {}
    for key, outs in zip(prompts.loc[is_free, "key"], [s for s, f in zip(samples, is_free) if f]):
        decisions = [parse_output(t, lang).decision for t in outs]
        free_right[str(key)] = sum(d == reference[str(key)] for d in decisions) / len(decisions)
    attributed = [s for s, f in zip(samples, is_free) if not f]
    wrong = [sum(parse_output(t, lang).decision == OP.flip(ref) for t in outs)
             for outs, ref in zip(attributed, rows["reference_decision"])]
    out = rows[["pair_id", "protected_group", "protected_attr", "condition",
                "reference_decision"]].copy()
    out["n_wrong"] = wrong
    out["free_right_rate"] = [free_right[str(p)] for p in rows["pair_id"]]
    out["tipped_wrong"] = (out["free_right_rate"] > min_free_right) & (out["n_wrong"] > 0)
    return out


def paired_difference(base: pd.DataFrame, new: pd.DataFrame, boot: int, seed: int) -> dict:
    """Rates of tipped_wrong in each run and new - base, with a bootstrap over pairs."""
    pairs = np.array(sorted(set(base["pair_id"]) & set(new["pair_id"])))
    def sums(frame):
        g = frame.groupby("pair_id")["tipped_wrong"].agg(["sum", "size"])
        g = g.reindex(pairs, fill_value=0)
        return g["sum"].to_numpy(float), g["size"].to_numpy(float)
    b_hit, b_n = sums(base)
    n_hit, n_n = sums(new)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        bn, nn = b_n[idx].sum(), n_n[idx].sum()
        if bn and nn:
            draws.append(n_hit[idx].sum() / nn - b_hit[idx].sum() / bn)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
    rate = lambda h, n: float(h.sum() / n.sum()) if n.sum() else None  # noqa: E731
    return {"pairs": len(pairs), "base_rows": int(b_n.sum()), "base_tipped_wrong": int(b_hit.sum()),
            "base_rate": rate(b_hit, b_n), "new_rows": int(n_n.sum()),
            "new_tipped_wrong": int(n_hit.sum()), "new_rate": rate(n_hit, n_n),
            "difference": (rate(n_hit, n_n) - rate(b_hit, b_n))
            if b_n.sum() and n_n.sum() else None,
            "ci95": [float(lo), float(hi)], "bootstrap": len(draws)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()

    base = row_stats(Path(resolve_output_path(args.base)), args.chunk_size)
    new = row_stats(Path(resolve_output_path(args.new)), args.chunk_size)
    base = base[base["protected_group"] == args.group]
    shared = set(base["pair_id"]) & set(new["pair_id"])
    base, new = base[base["pair_id"].isin(shared)], new[new["pair_id"].isin(shared)]

    summary = {"base": args.base, "new": args.new, "group": args.group,
               "overall": paired_difference(base, new, args.boot, args.seed)}
    for column in ("condition", "reference_decision"):
        summary[f"by_{column}"] = {
            str(v): paired_difference(base[base[column] == v], new[new[column] == v],
                                      args.boot, args.seed)
            for v in sorted(set(base[column]) | set(new[column]))}
    err_b = base.groupby("pair_id")["tipped_wrong"].sum()
    err_n = new.groupby("pair_id")["tipped_wrong"].sum()
    pairs = pd.DataFrame({"base_error_rows": err_b, "new_error_rows": err_n}).fillna(0).astype(int)
    pairs = pairs[(pairs["base_error_rows"] > 0) | (pairs["new_error_rows"] > 0)]
    summary["erring_pairs"] = {
        "base": int((pairs["base_error_rows"] > 0).sum()),
        "new": int((pairs["new_error_rows"] > 0).sum()),
        "both": int(((pairs["base_error_rows"] > 0) & (pairs["new_error_rows"] > 0)).sum()),
        "new_error_rows_on_pairs_erring_in_base": int(
            pairs.loc[pairs["base_error_rows"] > 0, "new_error_rows"].sum()),
    }
    by_attr = (new.groupby("protected_attr")["tipped_wrong"].agg(new_rows="size",
                                                                 new_tipped_wrong="sum")
               .join(base.groupby("protected_attr")["tipped_wrong"]
                     .agg(base_rows="size", base_tipped_wrong="sum"), how="outer")
               .fillna(0).astype(int))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    by_attr.to_csv(out / "by_attribute.csv")
    pairs.sort_values("new_error_rows", ascending=False).to_csv(out / "pairs.csv")
    print(json.dumps(summary["overall"], indent=1))
    print(json.dumps(summary["erring_pairs"], indent=1))


if __name__ == "__main__":
    main()
