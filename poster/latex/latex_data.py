"""Numbers for the LaTeX poster, written into ``main.tex`` between its GENERATED DATA markers.

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
import json
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
HF_V3_REVISION = "3b8b1c47189d00ca9cf4eb88195c0b84ff81d889"   # hf_v3_manifest.json was taken here
SFT_V3 = "SFT-v3"

# slot, recipe or arm label (None: not trained), colour, filled, short name
SLOTS = [
    ("Base", None, "gray", 0, "Base"),
    ("Best SFT (SFT-v3)", SFT_V3, "blue", 0, "SFT"),
    ("Best DPO", "DPO_V3_4880", "amber", 0, "DPO"),
    ("Best GRPO", "GRPO+decay R4", "purple", 0, "GRPO"),
    ("Distilled SFT", "Distill", "blue", 1, "dSFT"),
    ("Distilled DPO (n=1)", "Distill on DPO", "amber", 1, "dDPO"),
    ("Distilled GRPO (n=1)", "Distill on GRPO R4", "purple", 1, "dGRPO"),
]
# The students (docs/PATCHING_DISTILL.md, B): one seed each, distilled from this parent arm.
STUDENT_PARENT = {"Distill on DPO": "DPO_V3_4880 s42", "Distill on GRPO R4": "GRPO+decay R4 s42"}
# The reports/cost/params.csv row that counts each trained slot's LoRA (one stage).
PARAMS_ROW = {SFT_V3: "qwen3.5-9b_en_only_sft_v3_additional",
              "DPO_V3_4880": "qwen3.5-9b_en_only_dpo_v3_4880",
              "GRPO+decay R4": "grpo_improve_r4_tipped_v2",
              "Distill": "decision_distill_seed42",
              "Distill on DPO": "decision_distill_dpo4880_seed42",
              "Distill on GRPO R4": "decision_distill_grpo_r4_seed42"}


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

    def parent_counts(label: str) -> pd.DataFrame:
        run = arms[label]
        f, r = counts(run), refm[run]
        u, a = 100 * f.unstable.sum() / f.sets.sum(), 100 * f.agree.sum() / f.decided.sum()
        if abs(u - float(r["unstable_within_group_pct"])) > 0.01 or \
           abs(a - float(r["agree_gpt4o_pct"])) > 0.01:
            raise SystemExit(f"{run}: {u:.3f}/{a:.3f} differ from reference_metrics")
        return f.reindex(index, fill_value=0.0)

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
                                   short=short, seeds=0, hasparent=0, pu=0, pq=0))
            continue
        u = ratio(per[slot], "unstable", "sets", w)
        q = ratio(per[slot], "agree", "decided", w)
        runs = runs_of(name)
        if name is None:
            p, data, train, dist = "--", 0.0, 0.0, 0.0
        else:
            p = f"{int(params[PARAMS_ROW[name]]['params_trainable']) / 1e6:.1f}*"   # footnoted
            if name == SFT_V3:
                data, train, dist = sft_data, sft_train, 0.0
            elif name in STUDENT_PARENT:
                # final_table's train adds the parent's training: split it back out. The
                # student's data_jobs list its parent's too (reports/final/arms.csv).
                parent = table[STUDENT_PARENT[name]]
                data = sft_data + mean_of(name, "data_gpu_h")
                train = sft_train + float(parent["train_gpu_h"])
                dist = mean_of(name, "train_gpu_h") - float(parent["train_gpu_h"])
            elif recipes[name]["family"] == "distill":
                data = sft_data + mean_of(name, "data_gpu_h")
                train, dist = sft_train, mean_of(name, "train_gpu_h")
            else:
                data = sft_data + mean_of(name, "data_gpu_h")
                train, dist = sft_train + mean_of(name, "train_gpu_h"), 0.0
        # A student's own parent (one seed, the best of three), drawn open on the student's row:
        # the Best DPO / Best GRPO rows are three-seed means, not what the student started from.
        pu = pq = 0.0
        if name in STUDENT_PARENT:
            parent_per = parent_counts(STUDENT_PARENT[name])
            pu = ratio(parent_per, "unstable", "sets", w)[0]
            pq = ratio(parent_per, "agree", "decided", w)[0]
        model_rows.append(dict(
            model=slot, params=p, colour=colour, filled=filled, pending=0,
            instability=round(u[0], 2), ilo=round(u[1], 2), ihi=round(u[2], 2),
            quality=round(q[0], 2), qlo=round(q[1], 2), qhi=round(q[2], 2),
            data=round(data, 2), patch=0, train=round(train, 2), distill=round(dist, 2),
            eval=round(eval_gpu_h, 2), short=short, seeds=len(runs),
            hasparent=int(name in STUDENT_PARENT), pu=round(pu, 2), pq=round(pq, 2)))

    # Paired change in agreement, distilled minus parent: SFT-v3 for the distilled SFT, and each
    # student's own seed-42 parent, not its method's three-seed mean.
    delta_rows = []
    for method, colour, student, parent in (
            ("SFT", "blue", "Distilled SFT", "Best SFT (SFT-v3)"),
            ("DPO", "amber", "Distilled DPO (n=1)", STUDENT_PARENT["Distill on DPO"]),
            ("GRPO", "purple", "Distilled GRPO (n=1)", STUDENT_PARENT["Distill on GRPO R4"])):
        a = per[student]
        b = per[parent] if parent in per else parent_counts(parent)
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

    def interval(run: str, against: str) -> str:
        c = comp[(run, against)]
        return f"{tm(c['delta_pp'])} [{tm(c['ci_low_pp'])}, {tm(c['ci_high_pp'])}]"

    def per_seed(run_of, against_of) -> list[float]:
        return [float(comp[(run_of(s), against_of(s))]["delta_pp"]) for s in (42, 43, 44)]

    full = lambda s: arms[f"Distill s{s}"]  # noqa: E731
    window_only = per_seed(lambda s: arms[f"Distill layers 15-20 s{s}"], full)
    window_removed = per_seed(lambda s: f"{full(s)}_xw15_20", full)   # docs/PATCHING_DISTILL.md, F
    widen = pd.read_csv(ROOT / "reports/margin_readout/readout.csv")
    widen = widen[widen["label"].str.match(r"(DPO_V3_4880|GRPO\+decay R4) s4[234]$")]
    assert len(widen) == 6, widen["label"].tolist()
    window_params = int(params["decision_distill_l15_20_seed42"]["params_trainable"])
    student = {k: comp[(arms[f"{k} s42"], arms[STUDENT_PARENT[k]])] for k in STUDENT_PARENT}
    lp = params["decision_distill_l0_20_seed43"]
    d = recipes["Distill"]
    dseeds = [table[lab] for lab in labels_of("Distill")]
    raw = pd.read_parquet(RAW / f"{arms[SFT_V3]}.parquet",
                          columns=[*SET, "protected_attr", "candidate_id", "job_id"])
    att = raw[raw["condition"] != "attr_free"]
    values = att.groupby("protected_group")["protected_attr"].nunique()
    findings_data = {
        "NPairs": f"{raw['pair_id'].nunique():,}",
        "NCandidates": f"{raw['candidate_id'].nunique():,}",
        "NPrompts": f"{len(raw):,}",
        "NSets": f"{att.groupby(SET).ngroups:,}",
        "NGroups": str(len(values)),
        "NMilitary": str(values["military_status"]),
        "NReligion": str(values["religion"]),
        "NGender": str(values["gender"]),
        "BackboneB": f"{int(params['decision_distill_seed42']['params_base_text']) / 1e9:.2f}",
    }
    # The released training data (Stereotypes-in-LLMs/hiring-bias-mitigation-v3): its manifest,
    # pinned at the Hub revision in HF_V3_REVISION, copied to hf_v3_manifest.json.
    rel = json.loads((HERE / "hf_v3_manifest.json").read_text())["files"]
    findings_data.update({
        "RelSFT": f"{rel['data/sft/train.parquet']['rows']:,}",
        "RelDPO": f"{rel['data/dpo/train.parquet']['rows']:,}",
        "RelRL": f"{rel['data/rl/train.parquet']['rows']:,}",
    })
    findings = {**findings_data,
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
        "WindowParams": f"{window_params / 1e6:.1f}",
        "WindowOnly": tm(sum(window_only) / 3),
        "WindowOnlySeeds": " / ".join(tm(v) for v in window_only),
        "WindowRemoved": tm(sum(window_removed) / 3),
        "WindowRemovedSeeds": " / ".join(tm(v) for v in window_removed),
        "WidenRange": f"{widen['a_median'].min():.1f}--{widen['a_median'].max():.1f}",
        "StudentDPO": interval(arms["Distill on DPO s42"], arms["DPO_V3_4880 s42"]),
        "StudentGRPO": interval(arms["Distill on GRPO R4 s42"], arms["GRPO+decay R4 s42"]),
        "StudentDPOPoint": tm(student["Distill on DPO"]["delta_pp"]),
        "StudentGRPOPoint": tm(student["Distill on GRPO R4"]["delta_pp"]),
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
    # Written into main.tex between its markers, so the poster is one self-contained file.
    main_tex = HERE / "main.tex"
    tex = main_tex.read_text()
    begin, end = "% BEGIN GENERATED DATA\n", "% END GENERATED DATA\n"
    head, rest = tex.split(begin, 1)
    _, tail = rest.split(end, 1)
    main_tex.write_text(head + begin + "\n".join(out) + "\n" + end + tail)
    print("\n".join(out))


if __name__ == "__main__":
    main()
