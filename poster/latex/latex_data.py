"""Numbers for the LaTeX poster, written into ``main.tex`` between its GENERATED DATA markers.

The poster's five model slots: the base model, SFT-v3 (every method's start and the standard's
reference arm) and one stage after it: the best DPO, the best GRPO and self-distillation, three
seeds each. "Best" is the recipe with the lowest seed-mean change against SFT-v3 among those
with three training seeds (reports/final/recipes.csv); it was chosen after the audits.

Beside the audits, the patching of one seed per method against SFT-v3 (reports/interp/distill/
<spec>/, docs/RL_PATCHING.md, docs/PATCHING_DISTILL.md section 4, docs/DISTILL_PATCHING.md),
the overlap of each method's unstable sets across its seeds (reports/seed_overlap/overlap.csv)
and the margin scale per seed (reports/margin_readout/readout.csv) fill the PathData table.

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

# slot, recipe or arm label (None: the base model), colour, filled, short name. The poster has
# one neutral family and one accent (main.tex, DESIGN): Base gray and SFT-v3 ink, both open;
# every stage after SFT-v3 green and filled. Names and weight tell the rows apart.
SLOTS = [
    ("Base", None, "gray", 0, "Base"),
    ("SFT-v3", SFT_V3, "ink", 0, "SFT-v3"),
    ("DPO", "DPO_V3_4880", "greenDark", 1, "DPO"),
    ("GRPO + lr decay", "GRPO+decay R4", "greenDark", 1, "GRPO"),
    ("Self-distillation", "Distill", "greenDark", 1, "dSFT"),
]
POST_SFT = [slot for slot, name, *_ in SLOTS if name not in (None, SFT_V3)]
# The patched seed of each post-SFT method (reports/interp/distill/<spec>/), its seed-overlap
# recipe (reports/seed_overlap/overlap.csv) and its margin-readout label prefix.
PATCHED = {"DPO": ("dpo4880", "DPO_V3_4880", "DPO_V3_4880"),
           "GRPO + lr decay": ("grpo_r4", "GRPO+decay R4", "GRPO+decay R4"),
           "Self-distillation": ("distill43", "Distill", "distill")}
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

    model_rows = []
    for slot, name, colour, filled, short in SLOTS:
        u = ratio(per[slot], "unstable", "sets", w)
        q = ratio(per[slot], "agree", "decided", w)
        if name is None:
            p, data, train, post = "--", 0.0, 0.0, 0.0
        elif name == SFT_V3:
            p = f"{int(params[PARAMS_ROW[name]]['params_trainable']) / 1e6:.1f}*"   # footnoted
            data, train, post = sft_data, sft_train, 0.0
        else:
            # One stage on SFT-v3 merged: its own data beyond SFT-v3's and its own training
            # (final_table.csv data_gpu_h and train_gpu_h of the recipe, seed means).
            p = f"{int(params[PARAMS_ROW[name]]['params_trainable']) / 1e6:.1f}*"
            data = sft_data + mean_of(name, "data_gpu_h")
            train, post = sft_train, mean_of(name, "train_gpu_h")
        model_rows.append(dict(
            model=slot, params=p, colour=colour, filled=filled, pending=0,
            instability=round(u[0], 2), ilo=round(u[1], 2), ihi=round(u[2], 2),
            quality=round(q[0], 2), qlo=round(q[1], 2), qhi=round(q[2], 2),
            data=round(data, 2), train=round(train, 2), post=round(post, 2),
            eval=round(eval_gpu_h, 2), short=short, seeds=len(runs_of(name))))

    # Paired change in agreement with GPT-4o, each post-SFT method minus SFT-v3: the same
    # resampled candidates on both sides, seed-mean counts for the method.
    delta_rows = []
    for slot in POST_SFT:
        a, b = per[slot], per["SFT-v3"]
        point = 100 * (a.agree.sum() / a.decided.sum() - b.agree.sum() / b.decided.sum())
        draws = 100 * ((w @ a.agree.to_numpy()) / (w @ a.decided.to_numpy())
                       - (w @ b.agree.to_numpy()) / (w @ b.decided.to_numpy()))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        delta_rows.append(dict(method=slot, colour="greenDark", pending=0, delta=round(point, 2),
                               lo=round(lo, 2), hi=round(hi, 2)))

    # Matched-hire-rate effects against SFT-v3 (the standard), per seed and as seed means.
    def tm3(x: float | str) -> str:
        v = Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)
        return ("\\textminus{}" if v < 0 else "+") + f"{abs(v)}"

    def tm(x: float | str) -> str:
        """Signed, half to even on the value as written (as the cards print it)."""
        v = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        return ("\\textminus{}" if v < 0 else "+") + f"{abs(v)}"

    comp = {(r["run"], r["against"]): r for r in rows("reports/comparisons.csv")}

    def per_seed(run_of, against_of) -> list[float]:
        return [float(comp[(run_of(s), against_of(s))]["delta_pp"]) for s in (42, 43, 44)]

    def rng(vals, fmt: str) -> str:
        return f"{min(vals):{fmt}}--{max(vals):{fmt}}"

    full = lambda s: arms[f"Distill s{s}"]  # noqa: E731
    window_only = per_seed(lambda s: arms[f"Distill layers 15-20 s{s}"], full)
    window_removed = per_seed(lambda s: f"{full(s)}_xw15_20", full)   # docs/PATCHING_DISTILL.md, F
    window_kept = per_seed(lambda s: f"{full(s)}_xk15_20", full)      # docs/DISTILL_FOLLOWUP.md, H
    readout = pd.read_csv(ROOT / "reports/margin_readout/readout.csv")
    readout = readout.set_index("label")["a_median"]
    overlap = pd.read_csv(ROOT / "reports/seed_overlap/overlap.csv")
    # Rescaling SFT-v3's margins leaves the matched statistic unchanged (docs/SCALE_CONTROL.md, P1).
    sc = pd.read_csv(ROOT / "reports/scale_control/summary.csv")
    zero = (sc[[c for c in sc.columns if c.endswith("_delta")]] == 0).all().all()

    # One row per post-SFT method: the patched seed's route and tail against SFT-v3, the margin
    # scale over its three seeds, and how much its seeds' unstable sets overlap.
    # The head that leads DPO on implicit combat, where L15.H11 drops to rank 6.
    dpo_impl = pd.read_csv(ROOT / "reports/interp/distill/dpo4880/component_summary.csv")
    dpo_late = dpo_impl.set_index("contrast").loc["mil_combat_civ_impl", "top3_heads_spec"]
    dpo_late = dpo_late.split()[0]
    path_rows, scales, tails, jacs, borders, profiles, ranks = [], {}, {}, {}, {}, {}, {}
    n_contrasts: set[int] = set()
    for slot, short in ((s, sh) for s, n, _c, _f, sh in SLOTS if s in POST_SFT):
        spec, recipe, prefix = PATCHED[slot]
        sens = pd.read_csv(ROOT / f"reports/interp/distill/{spec}/sensitivity.csv")
        comps = pd.read_csv(ROOT / f"reports/interp/distill/{spec}/component_summary.csv")
        assert len(sens) == len(comps) >= 3, spec
        n_contrasts.add(len(sens))
        a_med = [float(readout[f"{prefix} s{s}"]) for s in (42, 43, 44)]
        jac = overlap[(overlap["recipe"] == recipe) & overlap["seeds"].str.contains("/")]["jaccard"]
        border = overlap[(overlap["recipe"] == recipe) & ~overlap["seeds"].str.contains("/")]
        border = 100 * border["on_borderline_share"]
        assert len(jac) == 3 and len(border) == 3, recipe
        p99 = sens["r_attr_p99"]
        scales[slot], tails[slot] = rng(a_med, ".2f"), rng(p99, ".1f" if p99.max() >= 1 else ".2f")
        jacs[slot], borders[slot] = rng(jac, ".2f"), rng(border, ".0f") + "\\%"
        profiles[slot] = rng(comps["profile_r"], ".2f")
        ranks[slot] = " / ".join(str(int(v)) for v in comps["l15h11_rank_spec"])
        same = (comps["profile_r"] >= 0.80).all() and (comps["l15h11_rank_spec"] <= 3).all()
        route = ("similar (attenuated)" if same and float(sens["r_scale"].iloc[0]) < 1
                 else "similar profile" if same else f"partly later ({dpo_late})")
        path_rows.append(dict(
            method=slot, short=short, rmean=round(float(comps["profile_r"].mean()), 3),
            rrange=profiles[slot], ranks=ranks[slot], scale=f"{float(sens['r_scale'].iloc[0]):.2f}",
            route=route, p99mean=round(float(sens["r_attr_p99"].mean()), 2), p99range=tails[slot],
            p99min=round(float(p99.min()), 2), p99max=round(float(p99.max()), 2),
            jmean=round(float(jac.mean()), 3), jrange=jacs[slot],
            jmin=round(float(jac.min()), 3), jmax=round(float(jac.max()), 3),
            border=borders[slot]))
    rl = [PATCHED[s][0] for s in ("DPO", "GRPO + lr decay")]
    rl_p99 = pd.concat(
        pd.read_csv(ROOT / f"reports/interp/distill/{spec}/sensitivity.csv")["r_attr_p99"]
        for spec in rl)
    rl_jac = overlap[overlap["recipe"].isin(["DPO_V3_4880", "GRPO+decay R4"])
                     & overlap["seeds"].str.contains("/")]["jaccard"]
    base_cmp = json.loads((HERE.parent / "base_compare.json").read_text())
    post = ("DPO_V3_4880", "GRPO+decay R4", "Distill")
    post_means = [float(recipes[n]["delta_common_pp"]) for n in post]
    post_gpu = [mean_of(n, "train_gpu_h") for n in post]

    seed_rows = []
    for slot, name, _c, _f, short in SLOTS:
        if slot not in POST_SFT:
            continue
        mean = float(recipes[name]["delta_common_pp"])
        for lab in labels_of(name):
            r = table[lab]
            seed_rows.append(dict(
                method=slot, short=short, seed=r["seed"],
                delta=round(float(r["delta_common_pp"]), 3),
                lo=round(float(r["ci_common_low"]), 3), hi=round(float(r["ci_common_high"]), 3),
                mean=round(mean, 3), meantxt=tm3(mean)))
    assert len(seed_rows) == 9, len(seed_rows)
    quality = {r["model"]: r["quality"] for r in model_rows}
    window_params = params["decision_distill_l15_20_seed42"]
    d = recipes["Distill"]
    dseeds = [table[lab] for lab in labels_of("Distill")]
    dpo_seeds = [table[lab] for lab in labels_of("DPO_V3_4880")]
    all_seeds = [table[lab] for n in post for lab in labels_of(n)]
    raw = pd.read_parquet(RAW / f"{arms[SFT_V3]}.parquet",
                          columns=[*SET, "protected_attr", "candidate_id", "job_id"])
    att = raw[raw["condition"] != "attr_free"]
    values = att.groupby("protected_group")["protected_attr"].nunique()
    findings = {
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
    findings.update({
        "RelSFT": f"{rel['data/sft/train.parquet']['rows']:,}",
        "RelDPO": f"{rel['data/dpo/train.parquet']['rows']:,}",
        "RelRL": f"{rel['data/rl/train.parquet']['rows']:,}",
    })
    # The poster quotes the common point: every arm moved to SFT-v3's own hire rate (the user,
    # 2026-11-01; docs/DISTILL_FOLLOWUP.md, section 0), not each arm's own point.
    agree = {r["method"]: r for r in delta_rows}
    findings.update({
        "CommonHire": f"{float(dseeds[0]['common_hire_rate_pct']):.2f}",
        "DistillMatched": tm(d["delta_common_pp"]),
        "DistillSeeds": " / ".join(tm(s["delta_common_pp"]) for s in dseeds),
        "DPOMatched": tm(recipes["DPO_V3_4880"]["delta_common_pp"]),
        "GRPOMatched": tm(recipes["GRPO+decay R4"]["delta_common_pp"]),
        # Seeds whose common-point interval lies below 0, of the nine post-SFT seeds.
        "CommonCIBelow": str(sum(float(s["ci_common_high"]) < 0 for s in all_seeds)),
        # DPO at its own point, the seed that read as a result and its replicates.
        "DPOSeedsOwn": " / ".join(tm(s["delta_own_pp"]) for s in dpo_seeds),
        "DPOBestOwn": tm(min(float(s["delta_own_pp"]) for s in dpo_seeds)),
        "AgreeDPO": tm(agree["DPO"]["delta"]),
        "AgreeGRPO": tm(agree["GRPO + lr decay"]["delta"]),
        "AgreeDistill": tm(agree["Self-distillation"]["delta"]),
        "AgreeAbsMax": f"{max(abs(float(r['delta'])) for r in delta_rows):.2f}",
        "PostDPO": f"{mean_of('DPO_V3_4880', 'train_gpu_h'):.1f}",
        "PostGRPO": f"{mean_of('GRPO+decay R4', 'train_gpu_h'):.1f}",
        "DistillTrainGPUh": f"{mean_of('Distill', 'train_gpu_h'):.1f}",
        "DistillExtraGPUh": f"{float(d['extra_gpu_h']):.1f}",
        "StandardParams": f"{int(params['decision_distill_seed42']['params_trainable']) / 1e6:.1f}",
        "StandardPct": f"{float(params['decision_distill_seed42']['trainable_pct_of_text']):.3f}",
        "WindowParams": f"{int(window_params['params_trainable']) / 1e6:.1f}",
        "WindowOnly": tm(sum(window_only) / 3),
        "WindowRemoved": tm(sum(window_removed) / 3),
        "WindowKept": tm(sum(window_kept) / 3),
        "WidenRange": rng([readout[f"{p} s{s}"] for p in ("DPO_V3_4880", "GRPO+decay R4")
                           for s in (42, 43, 44)], ".1f"),
        "ScaleControl": "exactly 0" if zero else "not 0",
        "SFTTrainGPUh": f"{sft_train:.1f}",
        "BaseAgree": f"{quality['Base']:.1f}",
        "SFTAgree": f"{quality['SFT-v3']:.1f}",
        # The take-home: Base moved to SFT-v3's hire rate against SFT-v3 (poster/base_compare.json,
        # operating_point_control.matched_comparison), and the three post-SFT means' span.
        "BaseAtSFT": f"{float(base_cmp['sft_v3']['base_unstable_pct']):.1f}",
        "SFTOwn": f"{float(base_cmp['sft_v3']['unstable_pct']):.1f}",
        "SFTvsBase": (f"{tm(base_cmp['sft_v3']['delta_pp'])} [{tm(base_cmp['sft_v3']['ci'][0])}, "
                      f"{tm(base_cmp['sft_v3']['ci'][1])}]"),
        "PostLow": tm(min(post_means)), "PostHigh": tm(max(post_means)),
        "PostGPUhLow": f"{min(post_gpu):.1f}", "PostGPUhHigh": f"{max(post_gpu):.1f}",
        "TailRL": rng(rl_p99, ".1f"), "JacRL": rng(rl_jac, ".2f"),
        "DPOLateHead": dpo_late,
        **{f"Scale{k}": scales[s] for k, s in (("DPO", "DPO"), ("GRPO", "GRPO + lr decay"),
                                                ("Distill", "Self-distillation"))},
        **{f"Tail{k}": tails[s] for k, s in (("DPO", "DPO"), ("GRPO", "GRPO + lr decay"),
                                              ("Distill", "Self-distillation"))},
        **{f"Jac{k}": jacs[s] for k, s in (("DPO", "DPO"), ("GRPO", "GRPO + lr decay"),
                                            ("Distill", "Self-distillation"))},
        **{f"Border{k}": borders[s] for k, s in (("DPO", "DPO"), ("GRPO", "GRPO + lr decay"),
                                                  ("Distill", "Self-distillation"))},
        **{f"Prof{k}": profiles[s] for k, s in (("DPO", "DPO"), ("GRPO", "GRPO + lr decay"),
                                                 ("Distill", "Self-distillation"))},
    })

    # One case before and after (poster/cases.json, pick_cases.py): military status, implicit
    # wording, the base model's hire and reject on one CV, SFT-v3's one decision, quoted as written.
    def tex(text: str) -> str:
        return (text.replace("\\", "/").replace("&", "\\&").replace("%", "\\%")
                .replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")
                .replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
                .replace("«", "``").replace("»", "''"))

    def clipped(sentence: str) -> str:
        """The sentence up to its first comma, ' and' or ' in', then an ellipsis."""
        cut = min((i for i in (sentence.find(","), sentence.find(" and "), sentence.find(" in "))
                   if i > 0), default=len(sentence))
        return tex(sentence[:cut].rstrip()) + " \\ldots"

    def cue(sentence: str, other: str) -> str:
        """The clipped sentence, the words it does not share with the other one emphasised."""
        words = clipped(sentence).split(" ")
        theirs = clipped(other).split(" ")
        assert len(words) == len(theirs), (words, theirs)
        out, faint = [], []
        for w, o in zip(words, theirs, strict=True):
            if w == o:
                faint.append(w)
                continue
            if faint:
                out.append(f"\\Faint{{{' '.join(faint)}}}")
                faint = []
            out.append(w)
        if faint:
            out.append(f"\\Faint{{{' '.join(faint)}}}")
        return " ".join(out)

    cases = json.loads((HERE.parent / "cases.json").read_text())
    case = next(c for c in cases["cases"] if c["group"] == cases["poster_case"])
    case_counts = cases["counts"]
    sft_reason = case["sft_v3"]["feedback"].split(". ")[0].rstrip(".") + "."
    def month(t: str) -> str:
        return f"{t[3:5]}/{t[6:]}"

    roles = []
    for i, pos in enumerate(case["cv_positions"]):
        a, b = re.split(r"\s*[-–]\s*", pos["dates"])
        duty = pos["detail"].split(" and ")[0].rstrip(". ")
        duty = f" ({tex(duty[:1].lower() + duty[1:])})" if i == 0 else ""
        roles.append(f"{tex(pos['position'])}{duty}, {month(a)}--{month(b)}")
    decisions = case["base"]["decisions"]
    rejected = [v for v, d in decisions.items() if d == "reject"]
    findings.update({
        "CaseJob": tex(case["job_position"]),
        "CaseRequirement": tex(case["job_requirement"].split(", account")[0]
                                  .replace("years' of", "years of")),
        # The CV's dated roles as written and the years they cover, overlaps counted once
        # (pick_cases.py, years_covered): the reader checks the requirement against the CV.
        "CaseRoles": "; ".join(roles),
        "CaseYears": f"{case['cv_years_covered']}",
        "CaseHireSentence": cue(case["base"]["hire"]["sentence"],
                                case["base"]["reject"]["sentence"]),
        "CaseRejectSentence": cue(case["base"]["reject"]["sentence"],
                                  case["base"]["hire"]["sentence"]),
        "CaseHireReason": tex(case["base"]["hire"]["feedback"]),
        "CaseRejectReason": tex(case["base"]["reject"]["feedback"]),
        "CaseSFTReason": tex(sft_reason),
        # Faithful excerpts for the poster: the reason up to its first ';' or ',', then an ellipsis.
        "CaseRejectExcerpt": tex(case["base"]["reject"]["feedback"].split(";")[0])
        + " \\ldots",
        "CaseSFTExcerpt": tex(sft_reason.split(",")[0]) + " \\ldots",
        "CaseGroup": case["group"].replace("_", " "),
        "CaseVariants": str(len(decisions)),
        "CaseBaseRejects": str(len(rejected)),
        "CaseRejectValues": tex(", ".join(rejected)),
        "CaseUnstableTotal": str(sum(c["base_unstable_sets"] for c in case_counts.values())),
        "CaseStableSFT": str(sum(c["stable_in_SFT-v3"] for c in case_counts.values())),
        "CaseStableDistill": str(sum(c["stable_in_Self-distillation s43"]
                                     for c in case_counts.values())),
    })

    # The NVIDIA stack: the audits' engine against the naive setup (docs/THROUGHPUT_B.md, round
    # 2), HF against vLLM decisions (reports/interp/margins_summary-*.json), the project's cost
    # (reports/cost/by_stage.csv), and the acceleration benchmarks of docs/ACCEL_BENCH.md, each
    # 'to run' until its files exist.
    full = pd.read_csv(ROOT / "reports/throughput_b/full/speedup_summary.csv")
    engine = full[(full["condition"] == "cacheon-asbuilt-nostats")
                  & (full["ref_condition"] == "cacheoff-shuffle")].iloc[0]
    agree = [json.loads((ROOT / f"reports/interp/margins_summary-{spec}-grid-en.json")
                        .read_text())["audit"]["agreement_pct"]
             for spec in ("sft_v3", "dpo4880", "grpo_r4", "distill43")]
    stage = pd.read_csv(ROOT / "reports/cost/by_stage.csv")
    findings.update({
        "AuditWallS": f"{float(engine['wall_s_median']):.0f}",
        "AuditGenS": f"{float(engine['generate_s_median']):.0f}",
        "AuditReqPerS": f"{float(engine['requests_per_s_median']):.1f}",
        "AuditKWh": f"{float(engine['gpu_energy_kwh_median']):.3f}",
        "PrefixSpeedup": f"{float(engine['speedup_x_median']):.2f}",
        "AgreeHFvLLM": rng(agree, ".1f"),
        "ProjectGPUh": f"{float(stage.set_index('stage').loc['total allocated', 'gpu_h']):.0f}",
        "ProjectKWh": f"{stage['gpu_energy_kwh'].sum():.0f}",
    })
    # The patching workload and its cost (reports/cost/steps.csv, stage patching): one `patch-*`
    # step is one (model, contrast) on one GH200; a full-grid `margins-*` step reads 31,050
    # prompts. Patched forwards per pair from scripts/interp/10_patch.py: residual at 32 layers
    # plus the final pre-norm (33) and components (32 mixers + 32 MLPs), each over two token
    # classes, then the top 24 heads per class and 8 random heads re-measured by real patching;
    # AtP over 128 attention and 768 GDN heads in one backward.
    cost_steps = pd.read_csv(ROOT / "reports/cost/steps.csv", dtype={"job_id": str})
    patching = cost_steps[(cost_steps["stage"] == "patching") & (cost_steps["source"] == "record")]
    ok = patching[patching["status"] == "ok"]
    patch_ok = ok[ok["label"].str.startswith("patch-")]
    # The medians are over 64-pair runs: the gender runs (45 pairs against SFT-v3) are left out.
    patch_64 = patch_ok[~patch_ok["label"].str.contains("gen_male_female")]
    margins_ok = ok[ok["label"].str.startswith("margins-") & (ok["wall_s"] > 600)]
    assert len(patch_ok) >= 10 and len(margins_ok) >= 5, (len(patch_ok), len(margins_ok))
    audits = [json.loads((ROOT / f"reports/interp/margins_summary-{spec}-grid-en.json")
                         .read_text())["audit"]
              for spec in ("sft_v3", "dpo4880", "grpo_r4", "distill43")]
    # Interventions per pair in the poster's patched runs (scripts/interp/10_patch.py): the
    # real-patching sweep (real-*.parquet) and the head verification (verify.parquet), per pair.
    evals, real_n, verify_n = [], [], []
    patch_root = RAW.parent / "interp/patching"
    for spec in ("sft_v3", *(v[0] for v in PATCHED.values())):
        for run in sorted((patch_root / spec / "grid").glob("*-n64-s42-sel_sft_v3")):
            if not (run / "verify.parquet").exists():  # a run still in progress
                continue
            real = pd.concat(pd.read_parquet(f, columns=["pair_uid"]) for f in run.glob("real-*"))
            ver = pd.read_parquet(run / "verify.parquet", columns=["pair_uid"])
            r, v = real.groupby("pair_uid").size(), ver.groupby("pair_uid").size()
            assert r.nunique() == 1 and v.nunique() == 1, run
            real_n.append(int(r.iloc[0]))
            verify_n.append(int(v.iloc[0]))
            evals.append(real_n[-1] + verify_n[-1])
    assert len(evals) >= 12, len(evals)
    backend_gap = max(abs(a["unstable_sets_hf_pct"] - a["unstable_sets_audit_pct"]) for a in audits)
    # The comparison 4A makes is arm minus SFT-v3; how far that difference moves between the
    # two backends (greedy unstable-set rate at each model's own hire rate, patched seeds).
    ref = audits[0]
    delta_shift = max(abs((a["unstable_sets_hf_pct"] - ref["unstable_sets_hf_pct"])
                          - (a["unstable_sets_audit_pct"] - ref["unstable_sets_audit_pct"]))
                      for a in audits[1:])
    findings.update({
        "NPatchContrasts": str(max(n_contrasts)),
        "PatchEvalsPerPair": rng(evals, "d"),
        "PatchRealPerPair": rng(real_n, "d"),
        "PatchVerifyPerPair": rng(verify_n, "d"),
        "AtPHeads": f"{128 + 768:,}",
        "PatchStepMin": f"{patch_64['wall_s'].median() / 60:.0f}",
        "PatchStepGPUh": f"{patch_64['gpu_h'].median():.2f}",
        "PatchRuns64": str(len(patch_64)),
        "PatchPeakGiB": f"{patch_ok['mem_peak_gib'].max():.0f}",
        "PatchStepsOK": str(len(patch_ok)),
        "MarginsStepMin": f"{margins_ok['wall_s'].median() / 60:.0f}",
        "MarginsPeakGiB": f"{margins_ok['mem_peak_gib'].max():.0f}",
        "PatchTotalGPUh": f"{float(stage.set_index('stage').loc['patching', 'gpu_h']):.1f}",
        "BackendUnstableMax": f"{backend_gap:.2f}",
        "BackendDeltaShift": f"{delta_shift:.2f}",
    })
    accel_rows = []
    speed = ROOT / "reports/throughput_b/accel/speedup_summary.csv"
    b1 = b1_times = None
    if speed.exists():
        t = pd.read_csv(speed)
        hit = t[(t["condition"] == "vllm") & (t["ref_condition"] == "hf8")]
        b1 = float(hit["speedup_x_median"].iloc[0]) if len(hit) else None
        if len(hit):
            b1_times = (float(hit["ref_generate_s_median"].iloc[0]),
                        float(hit["generate_s_median"].iloc[0]))
    steps = pd.read_csv(ROOT / "reports/cost/steps.csv", dtype={"job_id": str})
    steps = steps[(steps["source"] == "record") & (steps["status"] == "ok")]

    def runtime(label: str) -> float | None:
        hit = steps[steps["label"] == label]
        return float(hit["train_runtime_s"].iloc[-1]) if len(hit) else None

    def speedup_ratio(before: str, after: str) -> float | None:
        a, b = runtime(before), runtime(after)
        return a / b if a and b else None

    def secs(x: float) -> str:
        return f"{x:,.0f}".replace(",", "\\thinspace{}")  # for the detail macros, not tables

    tp = pd.read_csv(ROOT / "reports/cost/throughput.csv")
    tp = tp[tp["label"] == "throughput-accel-vllm"]
    st_b3 = pd.read_csv(ROOT / "reports/throughput_b/accel/step_times.csv")
    st_b3 = st_b3[st_b3["bench"] == "B3"].groupby("label")[["step_time_s",
                                                             "completion_tokens_mean"]].mean()
    if b1_times and len(tp):
        findings["AuditDetail"] = (
            f"HF generate \\textrightarrow{{}} vLLM: {secs(b1_times[0])} \\textrightarrow{{}} "
            f"{b1_times[1]:.1f} s, generation only; {int(tp['requests'].iloc[0]):,} prompts, "
            f"about {(tp['output_tokens'] / tp['requests']).mean():.0f} tokens out")
    findings["TrainDetail"] = (
        f"1 \\textrightarrow{{}} 4 GH200, global batch 32, same 40 steps: "
        f"{secs(runtime('bench-distill-ddp1'))} \\textrightarrow{{}} "
        f"{secs(runtime('bench-distill-ddp4'))} s")
    hf_s, vl_s = (st_b3.loc[f"bench-grpo-rollout-{k}", "step_time_s"] for k in ("hf", "vllm"))
    findings["GRPODetail"] = (
        f"HF \\textrightarrow{{}} colocated vLLM, 6 steps: "
        f"{secs(runtime('bench-grpo-rollout-hf'))} \\textrightarrow{{}} "
        f"{secs(runtime('bench-grpo-rollout-vllm'))} s; "
        f"{st_b3['completion_tokens_mean'].mean():.0f}-token outputs, so only "
        f"{hf_s - vl_s:.1f} of {hf_s:.1f} s a step is saved")
    for key, label, value in (
            ("audit", "Audit generation", b1),
            ("train", "Training", speedup_ratio("bench-distill-ddp1", "bench-distill-ddp4")),
            ("grpo", "GRPO rollouts",
             speedup_ratio("bench-grpo-rollout-hf", "bench-grpo-rollout-vllm"))):
        # Plain cells only: the label is formatted in main.tex from `speedup`.
        accel_rows.append(dict(key=key, label=label,
                               speedup=round(value, 2) if value else 0,
                               pending=int(value is None)))

    # Per-step times (scripts/accel_step_times.py): B2's two phases, since the 1-GPU run slowed
    # after step 5 and the ratio over train_runtime mixes them; B3's per-step saving and the
    # completions' length, which bound what a faster sampler can save.
    st = pd.read_csv(ROOT / "reports/throughput_b/accel/step_times.csv")
    b2 = st[st["bench"] == "B2"].pivot(index="step", columns="label", values="elapsed_s")
    one, four = b2["bench-distill-ddp1"], b2["bench-distill-ddp4"]
    b3 = st[st["bench"] == "B3"].groupby("label")[["step_time_s", "completion_tokens_mean"]].mean()
    hf, vl = b3.loc["bench-grpo-rollout-hf"], b3.loc["bench-grpo-rollout-vllm"]
    findings.update({
        "TrainOneEarly": f"{one[5] / 5:.0f}",
        "TrainOneLate": f"{(one[40] - one[5]) / 35:.0f}",
        "TrainRatioEarly": f"{(one[5] / 5) / (four[5] / 5):.1f}",
        "TrainRatioLate": f"{(one[40] - one[5]) / (four[40] - four[5]):.1f}",
        "RolloutTokens": f"{b3['completion_tokens_mean'].mean():.0f}",
        "RolloutSavedS": f"{hf['step_time_s'] - vl['step_time_s']:.1f}",
        "RolloutStepS": f"{hf['step_time_s']:.1f}",
    })

    def tex_table(name: str, recs: list[dict]) -> str:
        cols = list(recs[0])
        # col sep=comma: a comma inside a value would shift every later column of its row.
        bad = [(c, r[c]) for r in recs for c in cols if "," in str(r[c])]
        assert not bad, (name, bad)
        # Cells are read in the preamble: only the escapes the poster has compiled with.
        def rest(v: str) -> str:
            for ok in ("\\textminus{}", "\\texttimes", "\\%"):
                v = v.replace(ok, "")
            return v
        bad = [(c, r[c]) for r in recs for c in cols if "\\" in rest(str(r[c]))]
        assert not bad, (name, bad)
        body = "\n".join(",".join(str(r[c]) for c in cols) for r in recs)
        return f"\\pgfplotstableread[col sep=comma]{{\n{','.join(cols)}\n{body}\n}}\\{name}\n"

    out = ["% Generated by poster/latex/latex_data.py -- do not edit; rerun the script.",
           tex_table("ModelData", model_rows), tex_table("DeltaData", delta_rows),
           tex_table("PathData", path_rows), tex_table("SeedData", seed_rows),
           tex_table("AccelData", accel_rows)]
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
