"""Numbers for the LaTeX poster (GTC_Poster.tex), written to ``poster_data.tex``.

The poster's seven model slots: the base model, the best SFT, DPO and GRPO, and the distilled
version of each. "Best" is the recipe with the lowest seed-mean change against SFT-v3 among
those with three training seeds (reports/final/recipes.csv); it was chosen after the audits.
A slot with no trained model is marked pending, never filled in.

Per model, from the audits' greedy decisions (outputs/raw/<run>.parquet), as in
reports/reference_metrics.csv: the share of unstable within-group sets and the agreement with
GPT-4o's reference, each with a 95% CI from 1,000 resamples of candidates (seed 42). For a
recipe with several seeds the per-candidate counts are averaged over its seeds first. The
paired change in agreement, distilled minus parent, resamples the same candidates for both.
Costs come from reports/final/final_table.csv and eval_cost.csv; the matched-hire-rate effects
from final_table.csv and recipes.csv.

    HBM_OUTPUT_ROOT=$XW ../venv/bin/python poster/latex/latex_data.py
"""

from __future__ import annotations

import csv
import os
import re
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw")) / "outputs/raw"
SET = ["pair_id", "protected_group", "condition"]
N_BOOT, SEED = 1000, 42
BASE_RUN = "Qwen3.5-9B--en--baseline--cscs"
SFT_V3 = "SFT-v3"

# slot, recipe or arm label (None: not trained), colour, filled, short name
SLOTS = [
    ("Base", None, "gray", 0, "Base"),
    ("Best SFT (SFT-v3)", SFT_V3, "blue", 0, "SFT"),
    ("Best DPO", "DPO_V3_4880", "amber", 0, "DPO"),
    ("Best GRPO", "GRPO+decay R4", "purple", 0, "GRPO"),
    ("Distilled SFT", "Distill", "blue", 1, "dSFT"),
    ("Distilled DPO", None, "amber", 1, "dDPO"),
    ("Distilled GRPO", None, "purple", 1, "dGRPO"),
]
# The reports/cost/params.csv row that counts each trained slot's LoRA (one stage).
PARAMS_ROW = {SFT_V3: "qwen3.5-9b_en_only_sft_v3_additional",
              "DPO_V3_4880": "qwen3.5-9b_en_only_dpo_v3_4880",
              "GRPO+decay R4": "grpo_improve_r4_tipped_v2",
              "Distill": "decision_distill_seed42"}


def rows(rel: str) -> list[dict]:
    with open(ROOT / rel, newline="") as f:
        return list(csv.DictReader(f))


def counts(run: str) -> pd.DataFrame:
    """Per candidate: unstable and complete sets, agreeing and decided prompts."""
    df = pd.read_parquet(RAW / f"{run}.parquet",
                         columns=[*SET, "protected_attr", "decision", "outcome",
                                  "reference_decision"])
    df["cand"] = df["pair_id"].str.split("__").str[0]
    decided = df["outcome"] == "decided"
    ref = df["reference_decision"].astype(str).str.strip().str.lower()
    df["agree"] = np.where(decided, (df["decision"] == ref).astype(float), np.nan)
    att = df[df["condition"] != "attr_free"].assign(d=lambda x: x["decision"].where(decided))
    g = att.groupby(SET)["d"]
    sets = pd.DataFrame({"complete": g.count() == g.size(), "unstable": g.nunique() > 1})
    sets = sets[sets["complete"]].reset_index()
    sets["cand"] = sets["pair_id"].str.split("__").str[0]
    per = pd.DataFrame({
        "unstable": sets.groupby("cand")["unstable"].sum(),
        "sets": sets.groupby("cand").size(),
        "agree": df.groupby("cand")["agree"].sum(),
        "decided": df.groupby("cand")["agree"].count(),
    }).fillna(0.0)
    return per


def boot_weights(index: pd.Index) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.multinomial(len(index), np.full(len(index), 1 / len(index)), size=N_BOOT)


def ratio(per: pd.DataFrame, num: str, den: str, w: np.ndarray) -> tuple[float, float, float]:
    point = 100 * per[num].sum() / per[den].sum()
    draws = 100 * (w @ per[num].to_numpy()) / (w @ per[den].to_numpy())
    return point, *np.percentile(draws, [2.5, 97.5])


def main() -> None:
    arms = {r["label"]: r["run"] for r in rows("reports/final/arms.csv")}
    table = {r["label"]: r for r in rows("reports/final/final_table.csv")}
    recipes = {r["name"]: r for r in rows("reports/final/recipes.csv")}
    refm = {r["run"]: r for r in rows("reports/reference_metrics.csv")}
    params = {r["run"]: r for r in rows("reports/cost/params.csv")}
    eval_gpu_h = float(rows("reports/final/eval_cost.csv")[0]["gpu_h_measured_median"])

    def labels_of(name: str) -> list[str]:
        return [lab for lab in arms if lab == name or re.fullmatch(rf"{re.escape(name)} s\d+", lab)]

    def runs_of(name: str | None) -> list[str]:
        return [BASE_RUN] if name is None else [arms[lab] for lab in labels_of(name)]

    # Per-candidate counts for every trained slot, averaged over seeds.
    per = {}
    for slot, name, *_ in SLOTS:
        if name is None and slot != "Base":
            continue
        frames = [counts(run) for run in runs_of(name)]
        for run, f in zip(runs_of(name), frames):      # the same numbers as reference_metrics
            r = refm[run]
            u, a = 100 * f.unstable.sum() / f.sets.sum(), 100 * f.agree.sum() / f.decided.sum()
            if abs(u - float(r["unstable_within_group_pct"])) > 0.01 or \
               abs(a - float(r["agree_gpt4o_pct"])) > 0.01:
                raise SystemExit(f"{run}: {u:.3f}/{a:.3f} differ from reference_metrics")
        per[slot] = sum(frames) / len(frames)
    index = per["Base"].index
    for k in per:
        per[k] = per[k].reindex(index, fill_value=0.0)
    w = boot_weights(index)

    sft = table[SFT_V3]
    sft_data, sft_train = float(sft["data_gpu_h"]), float(sft["train_gpu_h"])

    def mean_of(name: str, col: str) -> float:
        labs = labels_of(name)
        return sum(float(table[lab][col]) for lab in labs) / len(labs)

    model_rows, findings = [], {}
    for slot, name, colour, filled, short in SLOTS:
        pending = name is None and slot != "Base"
        if pending:
            model_rows.append(dict(model=slot, params="--", colour=colour, filled=filled,
                                   pending=1, instability=0, ilo=0, ihi=0, quality=0, qlo=0,
                                   qhi=0, data=0, patch=0, train=0, distill=0, eval=0,
                                   short=short, seeds=0))
            continue
        u = ratio(per[slot], "unstable", "sets", w)
        q = ratio(per[slot], "agree", "decided", w)
        runs = runs_of(name)
        if name is None:
            p, data, train, dist = "--", 0.0, 0.0, 0.0
        else:
            p = f"{int(params[PARAMS_ROW[name]]['params_trainable']) / 1e6:.1f}"
            if name == SFT_V3:
                data, train, dist = sft_data, sft_train, 0.0
            elif recipes[name]["family"] == "distill":
                data = sft_data + mean_of(name, "data_gpu_h")
                train, dist = sft_train, mean_of(name, "train_gpu_h")
            else:
                data = sft_data + mean_of(name, "data_gpu_h")
                train, dist = sft_train + mean_of(name, "train_gpu_h"), 0.0
        model_rows.append(dict(
            model=slot, params=p, colour=colour, filled=filled, pending=0,
            instability=round(u[0], 2), ilo=round(u[1], 2), ihi=round(u[2], 2),
            quality=round(q[0], 2), qlo=round(q[1], 2), qhi=round(q[2], 2),
            data=round(data, 2), patch=0, train=round(train, 2), distill=round(dist, 2),
            eval=round(eval_gpu_h, 2), short=short, seeds=len(runs)))

    # Paired change in agreement, distilled minus parent; pending where no student exists.
    delta_rows = []
    for method, colour, student, parent in (("SFT", "blue", "Distilled SFT", "Best SFT (SFT-v3)"),
                                            ("DPO", "amber", None, "Best DPO"),
                                            ("GRPO", "purple", None, "Best GRPO")):
        if student is None:
            delta_rows.append(dict(method=method, colour=colour, pending=1, delta=0, lo=0, hi=0))
            continue
        a, b = per[student], per[parent]
        point = 100 * (a.agree.sum() / a.decided.sum() - b.agree.sum() / b.decided.sum())
        draws = 100 * ((w @ a.agree.to_numpy()) / (w @ a.decided.to_numpy())
                       - (w @ b.agree.to_numpy()) / (w @ b.decided.to_numpy()))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        delta_rows.append(dict(method=method, colour=colour, pending=0, delta=round(point, 2),
                               lo=round(lo, 2), hi=round(hi, 2)))

    # Matched-hire-rate effects against SFT-v3 (the standard), per seed and as seed means.
    def tm(x: float | str) -> str:
        """Signed, half to even on the value as written (as the cards print it)."""
        v = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        return ("\\textminus{}" if v < 0 else "+") + f"{abs(v)}"

    comp = {(r["run"], r["against"]): r for r in rows("reports/comparisons.csv")}
    layers = comp[(arms["Distill layers 0-20 s43"], arms["Distill s43"])]
    lp = params["decision_distill_l0_20_seed43"]
    d = recipes["Distill"]
    dseeds = [table[lab] for lab in labels_of("Distill")]
    findings = {
        "DistillMatched": tm(d["delta_own_pp"]),
        "DistillSeeds": " / ".join(tm(s["delta_own_pp"]) for s in dseeds),
        "DistillSeedsBelowZero": str(sum(float(s["delta_own_pp"]) < 0 for s in dseeds)),
        "DistillCIBelowZero": str(sum(float(s["ci_own_high"]) < 0 for s in dseeds)),
        "DPOMatched": tm(recipes["DPO_V3_4880"]["delta_own_pp"]),
        "GRPOMatched": tm(recipes["GRPO+decay R4"]["delta_own_pp"]),
        "DistillExtraGPUh": f"{float(d['extra_gpu_h']):.1f}",
        "DistillTrainGPUh": f"{mean_of('Distill', 'train_gpu_h'):.1f}",
        "LayersControl": (f"{tm(layers['delta_pp'])} "
                          f"[{tm(layers['ci_low_pp'])}, {tm(layers['ci_high_pp'])}]"),
        "LayersParams": f"{int(lp['params_trainable']) / 1e6:.1f}",
        "LayersPct": f"{float(lp['trainable_pct_of_text']):.3f}",
        "StandardParams": f"{int(params['decision_distill_seed42']['params_trainable']) / 1e6:.1f}",
        "StandardPct": f"{float(params['decision_distill_seed42']['trainable_pct_of_text']):.3f}",
        "DistillAgreeDelta": tm(delta_rows[0]["delta"]),
        "DistillAgreeCI": f"[{tm(delta_rows[0]['lo'])}, {tm(delta_rows[0]['hi'])}]",
    }

    def tex_table(name: str, recs: list[dict]) -> str:
        cols = list(recs[0])
        body = "\n".join(",".join(str(r[c]) for c in cols) for r in recs)
        return f"\\pgfplotstableread[col sep=comma]{{\n{','.join(cols)}\n{body}\n}}\\{name}\n"

    out = ["% Generated by poster/latex/latex_data.py -- do not edit; rerun the script.",
           tex_table("ModelData", model_rows), tex_table("DeltaData", delta_rows)]
    for k, v in findings.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (HERE / "poster_data.tex").write_text("\n".join(out) + "\n")
    print((HERE / "poster_data.tex").read_text())


if __name__ == "__main__":
    main()
