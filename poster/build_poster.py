"""Build the GTC 2027 poster draft (44 x 35 in).

Step 1 reads the repository's result tables and writes ``poster/numbers.json``, and
``charts.py`` draws the poster's charts from the same tables into ``poster/figures/``. The poster
source (``poster/poster.typ``) takes every number from that file, so no number is typed into
the layout. Numbers that exist only in a hand-written card (``docs/*.md``) are copied into
``DOC_NUMBERS`` below with the card they come from; check them against the card before print.

Step 2 compiles ``poster.typ`` with Typst, with the repository as the root, so figures are
included from ``reports/`` directly. ``--preview`` also writes a PNG of the whole page.

Needs ``typst`` (the Python package); on Daint it lives in ``$XW/.typst-venv``:

    $XW/.typst-venv/bin/python poster/build_poster.py --preview /path/to/preview.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

# Numbers that live only in hand-written cards. Each entry names its card.
DOC_NUMBERS = {
    "audit_pairs": ("450", "README.md; docs/EVAL_STANDARD.md"),
    "audit_prompts": ("31,050", "docs/EVAL_STANDARD.md"),
    "audit_sets": ("2,700", "docs/EVAL_STANDARD.md"),
    "audit_candidates": ("150", "docs/SYSTEM_OVERVIEW.md, The stages"),
    "holdout_candidates": ("300", "docs/SYSTEM_OVERVIEW.md, The stages"),
    "grpo_pilot_delta": ("−0.95 [−1.90, −0.05]", "docs/GRPO_PILOT.md"),
    "grpo_pilot_hire": ("28.0% → 19.9%", "docs/GRPO_PILOT.md"),
    "sft_v3_effect_removed": ("61–77%",
                              "docs/INTERP_SFT_V3.md, Patching (patching_results.ipynb §6)"),
    "pathway_layers": ("15–20", "docs/DISTILL_PATCHING.md; docs/INTERP_SFT_V3.md"),
    "borderline_fixed_share": ("83–87%", "docs/SCALE_CONTROL.md, Result"),
    "lora_null_adapters": ("seven adapters, four objectives",
                           "reports/MITIGATION_FINDINGS.md, lesson 4"),
    "hf_vllm_agreement": ("99.77–99.79%", "docs/RL_PATCHING.md, Result, Gate"),
    "lora": ("r32, α64", "docs/SYSTEM_OVERVIEW.md, Post-training"),
    "vllm_version": ("0.31", "docs/SYSTEM_OVERVIEW.md, Audit"),
    "teacher": ("Qwen3.5-122B-A10B", "docs/SYSTEM_OVERVIEW.md, Teacher"),
    "vote_draws": ("8", "docs/SYSTEM_OVERVIEW.md, Teacher"),
    "patch_gate": ("97%", "docs/SYSTEM_OVERVIEW.md, HF consistency and mechanism"),
    "patch_pairs": ("64", "docs/INTERP_SFT_V3.md, Patching"),
    "patch_contrasts": ("3", "docs/INTERP_SFT_V3.md, Patching"),
}

DETAILS_URL = ("https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/"
               "poster/DETAILS.md")


def rows(rel: str) -> list[dict]:
    with open(ROOT / rel, newline="") as f:
        return list(csv.DictReader(f))


def one(rel: str, **match) -> dict:
    hits = [r for r in rows(rel) if all(r[k] == v for k, v in match.items())]
    if len(hits) != 1:
        raise SystemExit(f"{rel}: expected one row for {match}, found {len(hits)}")
    return hits[0]


def rnd(x: float | str, digits: int) -> Decimal:
    """Round half to even on the value as written, so -0.445 prints as the cards print it."""
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)


def pp(x: float | str, digits: int = 2, sign: bool = True) -> str:
    """Signed number with a real minus sign, as printed on the poster."""
    v = rnd(x, digits)
    s = f"{v:+}" if sign else f"{v}"
    return s.replace("-", "−")


def ci(lo: float | str, hi: float | str) -> str:
    return f"[{pp(lo)}, {pp(hi)}]"


def sd(row: dict, key: str) -> str:
    """A seed spread in brackets, or nothing for a single seed."""
    return f" ({rnd(row[key], 2)})" if row[key] else ""


def numbers() -> dict:
    n: dict = {"sources": {}}

    def put(key, value, source):
        n[key] = value
        n["sources"][key] = source

    # Reference arm and the common operating point.
    ft = "reports/final/final_table.csv"
    ref = one(ft, label="SFT-v3")
    put("common_hire", f"{float(ref['common_hire_rate_pct']):.2f}%", ft)
    put("ref_unstable", f"{float(ref['unstable_own_pct']):.2f}%", ft)

    comp = "reports/comparisons.csv"
    sft = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"
    v1 = one(comp, run=sft, against="Qwen3.5-9B--en--sft--adapter--en_only_cscs")
    put("sft_v3_vs_v1", f"{pp(v1['delta_pp'])} {ci(v1['ci_low_pp'], v1['ci_high_pp'])}", comp)
    base = one(comp, run=sft, against="Qwen3.5-9B--en--sft--adapter--en_only_cscs--base")
    put("sft_v3_vs_base",
        f"{pp(base['delta_pp'])} {ci(base['ci_low_pp'], base['ci_high_pp'])}", comp)

    # Recipes with seeds (means and sd over training seeds).
    rec = {r["name"]: r for r in rows("reports/final/recipes.csv")}
    table = []
    for name, label, note in [
        ("Distill", "Decision distillation", ""),
        ("GRPO+decay R4", "GRPO + lr decay", ""),
        ("DPO_V3_4880", "DPO, 4,880 on-policy pairs", ""),
        ("DAPO+decay D1", "DAPO + lr decay", ""),
        ("GRPO-v3-all ckpt 525", "GRPO, checkpoint 525", "exploratory pick"),
    ]:
        r = rec[name]
        table.append({
            "label": label,
            "seeds": r["seeds"],
            "own": pp(r["delta_own_pp"]) + sd(r, "delta_own_sd"),
            "common": pp(r["delta_common_pp"]) + sd(r, "delta_common_sd"),
            "gpu_h": f"{rnd(r['extra_gpu_h'], 1)}",
            "passes": r["passes"] == "True",
            "note": note,
        })
    put("recipes", table, "reports/final/recipes.csv")
    put("distill_mean", pp(rec["Distill"]["delta_own_pp"]), "reports/final/recipes.csv")
    put("distill_sd", f"{float(rec['Distill']['delta_own_sd']):.2f}", "reports/final/recipes.csv")

    # DPO_V3_4880's seeds.
    dpo = [r for r in rows(ft) if r["recipe"] == "dpo_v3_4880"]
    dpo.sort(key=lambda r: int(r["seed"]))
    put("dpo_seeds", [{"seed": r["seed"], "own": pp(r["delta_own_pp"]),
                       "ci": ci(r["ci_own_low"], r["ci_own_high"])} for r in dpo], ft)
    held = sum(float(r["ci_own_high"]) < 0 for r in dpo)
    put("dpo_held", f"{held} of {len(dpo)}", ft)
    put("dpo_merge_kept", f"{float(dpo[0]['merge_kept_pct']):.1f}%", ft)

    # Curves: at how many of the grid's hire rates each recipe's seed mean is below SFT-v3.
    cv = "reports/final/curves.csv"
    by: dict = {}
    for r in rows(cv):
        by.setdefault(r["label"], {})[float(r["hire_rate_pct"])] = float(r["unstable_pct"])
    ref_curve = by["SFT-v3"]
    below = {}
    for key, name in (("dpo", "DPO_V3_4880"), ("distill", "Distill"), ("grpo", "GRPO+decay R4")):
        seeds = [lab for lab in by if lab.startswith(name + " s")]
        n_below = sum(sum(by[s][x] - ref_curve[x] for s in seeds) < 0 for x in ref_curve)
        below[key] = f"{n_below} of {len(ref_curve)}"
    put("curve_below", below, cv)

    # Cost.
    fc = "reports/final/family_cost.csv"
    fam = {r["family"]: r for r in rows(fc)}
    d, g = float(fam["DPO after SFT-v3"]["gpu_h"]), float(fam["GRPO/DAPO after SFT-v3"]["gpu_h"])
    put("explore_dpo", f"{d:.0f}", fc)
    put("explore_rl", f"{g:.0f}", fc)
    put("explore_distill", f"{float(fam['Decision distillation']['gpu_h']):.0f}", fc)
    put("explore_ratio", f"{g / d:.1f}×", fc)
    total = one("reports/cost/by_stage.csv", stage="total allocated")
    put("gpu_h_total", f"{float(total['gpu_h']):.0f}", "reports/cost/by_stage.csv")
    ev = rows("reports/final/eval_cost.csv")[0]
    put("eval_gpu_h", f"{float(ev['gpu_h_measured_median']):.2f}", "reports/final/eval_cost.csv")

    # Part B, round 2: the audits' engine is cache on, built order, stats off.
    sp = "reports/throughput_b/full/speedup_summary.csv"
    eng = one(sp, condition="cacheon-asbuilt-nostats", ref_condition="cacheoff-shuffle")
    put("b_speedup", f"{float(eng['speedup_x_median']):.2f}×", sp)
    put("b_saved", f"{100 * float(eng['time_saved_median']):.1f}%", sp)
    order = one(sp, condition="cacheon-prefix", ref_condition="cacheon-asbuilt-nostats")
    put("b_order", f"{100 * float(order['time_saved_median']):.1f}%", sp)
    hit_built = one(sp, condition="cacheon-asbuilt", ref_condition="cacheoff-shuffle")
    hit_prefix = one(sp, condition="cacheon-prefix", ref_condition="cacheoff-shuffle")
    put("b_hit_built", f"{100 * float(hit_built['prefix_cache_hit_rate_median']):.1f}%", sp)
    put("b_hit_prefix", f"{100 * float(hit_prefix['prefix_cache_hit_rate_median']):.1f}%", sp)
    sh = one("reports/throughput_b/shared_prefix_all450.csv", scope="job")
    put("b_block_bound", f"{100 * float(sh['skippable_share_whole_blocks']):.2f}%",
        "reports/throughput_b/shared_prefix_all450.csv")
    put("b_token_shared", f"{100 * float(sh['skippable_share_token_exact']):.0f}%",
        "reports/throughput_b/shared_prefix_all450.csv")
    put("b_block", f"{float(sh['block']):.0f}", "reports/throughput_b/shared_prefix_all450.csv")

    # Mechanism: what later training does to SFT-v3's pathway.
    for arm, key in (("dpo4880", "dpo_scale"), ("grpo_r4", "grpo_scale")):   # the poster's arms
        path = f"reports/interp/distill/{arm}/sensitivity.csv"
        scale = {x["r_scale"] for x in rows(path)}
        if len(scale) != 1:
            raise SystemExit(f"{path}: r_scale differs between contrasts")
        put(key, f"{float(scale.pop()):.1f}×", path)
    path = "reports/interp/distill/distill43/component_summary.csv"
    cs = rows(path)
    kept = [100 * float(r["total_kept"]) for r in cs]
    put("distill_kept", f"{min(kept):.0f}–{max(kept):.0f}%", path)
    put("distill_l15h11_rank",
        "1" if all(r["l15h11_rank_spec"] == "1" for r in cs) else "varies", path)
    sc = rows("reports/scale_control/summary.csv")
    zero = all(float(r[k]) == 0.0 for r in sc
               for k in ("S_exact_a_median_delta", "S_q_a_median_delta",
                         "S_exact_a_slope_delta", "S_q_a_slope_delta"))
    put("scale_change", "exactly 0" if zero else "not 0", "reports/scale_control/summary.csv")
    es = json.loads((HERE / "effect_share.json").read_text())
    put("sft_v3_effect_removed_all", es["range"], "poster/effect_share.json (" + es["source"] + ")")
    mr = "reports/margin_readout/readout.csv"
    widen = [float(r["a_median"]) for r in rows(mr)
             if re.fullmatch(r"(DPO_V3_4880|GRPO\+decay R4) s4[234]", r["label"])]
    if len(widen) != 6:
        raise SystemExit(f"{mr}: expected six DPO and GRPO seeds, found {len(widen)}")
    put("widen_range", f"×{min(widen):.1f}–{max(widen):.1f}", mr)

    # The patching-informed round (docs/PATCHING_DISTILL.md), each arm head to head at a matched
    # hire rate (compare-arms rows of reports/comparisons.csv).
    arm_run = {r["label"]: r["run"] for r in rows("reports/final/arms.csv")}

    def vs(label: str, against: str) -> dict:
        return one(comp, run=arm_run.get(label, label), against=arm_run.get(against, against))

    def with_ci(r: dict) -> str:
        return f"{pp(r['delta_pp'])} {ci(r['ci_low_pp'], r['ci_high_pp'])}"

    def three(run_of) -> list[str]:
        ds = [float(vs(run_of(s), f"Distill s{s}")["delta_pp"]) for s in (42, 43, 44)]
        return [pp(sum(ds) / 3), " / ".join(pp(d) for d in ds)]

    put("window_only", three(lambda s: f"Distill layers 15-20 s{s}"), comp)
    put("window_removed", three(lambda s: arm_run[f"Distill s{s}"] + "_xw15_20"), comp)
    put("quarter_window", with_ci(vs("Distill window + GDN s42", "Distill s42")), comp)
    put("quarter_rank8", with_ci(vs("Distill rank 8 s42", "Distill s42")), comp)
    def students(student: str, parent: str) -> str:
        """Seeds 42 and 43, each against its own parent seed (docs/DISTILL_FOLLOWUP.md, G)."""
        rs = [vs(f"{student} s{s}", f"{parent} s{s}") for s in (42, 43)]
        mean = sum(float(r["delta_pp"]) for r in rs) / 2
        return f"{pp(mean)} (" + "; ".join(f"s{s} {with_ci(r)}" for s, r in zip((42, 43), rs)) + ")"

    put("student_dpo", students("Distill on DPO", "DPO_V3_4880"), comp)
    put("student_grpo", students("Distill on GRPO R4", "GRPO+decay R4"), comp)
    # docs/DISTILL_FOLLOWUP.md: H, the window kept alone, and I, two epochs, each against the
    # one-epoch all-layer adapter of the same seed.
    put("window_kept", three(lambda s: arm_run[f"Distill s{s}"] + "_xk15_20"), comp)
    put("two_epochs", three(lambda s: f"Distill two epochs s{s}"), comp)
    floor = [r for r in rows(comp) if r["run"].endswith("decision_distill_floor4_seed42")
             and r["against"] == arm_run["Distill s42"]]
    put("floor_pilot", with_ci(floor[0]) if floor else "", comp)
    params = {r["run"]: r for r in rows("reports/cost/params.csv")}
    for key, run in (("window_params", "decision_distill_l15_20_seed42"),
                     ("quarter_window_params", "decision_distill_w15_20_seed42"),
                     ("quarter_rank8_params", "decision_distill_r8_seed42")):
        put(key, f"{int(params[run]['params_trainable']) / 1e6:.1f}M", "reports/cost/params.csv")

    std = "src/hiring_bias_mitigation/analysis/eval_standard.py"
    code = (ROOT / std).read_text()
    for key in ("min_merge_kept_pct", "max_tpr_drop_pp", "max_agree_drop_pp", "max_undecided_pct"):
        m = re.search(rf"{key}: float = ([0-9.]+)", code)
        if not m:
            raise SystemExit(f"{std}: no default for {key}")
        put(key, f"{float(m.group(1)):g}", std)
    m = re.search(r"_step(\d+)$", dpo[0]["run"])
    put("dpo_steps", m.group(1) if m else "?", ft)

    # Results: the base model, SFT-v3 and the best recipe of each method (poster/base_compare.py).
    bc = json.loads((HERE / "base_compare.json").read_text())
    vs_base = [
        {"model": "Qwen3.5-9B (base)", "hire": f"{rnd(bc['base']['hire_pct'], 1)}%",
         "unstable": f"{rnd(bc['base']['unstable_pct'], 1)}%", "delta": "", "range": "",
         "gpu_h": "", "seeds": ""},
        {"model": "SFT-v3", "hire": f"{rnd(bc['sft_v3']['hire_pct'], 1)}%",
         "unstable": f"{rnd(bc['sft_v3']['unstable_pct'], 1)}%",
         "delta": pp(bc["sft_v3"]["delta_pp"]), "range": ci(*bc["sft_v3"]["ci"]),
         "gpu_h": f"{rnd(ref['extra_gpu_h'], 1)}", "seeds": "1"},
    ]
    for r in bc["rows"]:
        ds = [s["delta_pp"] for s in r["seeds"]]
        vs_base.append({
            "model": r["label"],
            "hire": f"{rnd(r['hire_pct_mean'], 1)}%",
            "unstable": f"{rnd(r['unstable_pct_mean'], 1)}%",
            "delta": pp(r["delta_pp_mean"]),
            "range": (f"{pp(min(ds))} to {pp(max(ds))}" if len(ds) > 1
                      else ci(*r["seeds"][0]["ci"])),
            "gpu_h": f"+{rnd(r['extra_gpu_h'], 1)}", "seeds": str(len(ds))})
    put("vs_base", vs_base, "poster/base_compare.json; reports/final/final_table.csv (GPU-h)")
    put("base_at_common", f"{rnd(bc['sft_v3']['base_unstable_pct'], 1)}%",
        "poster/base_compare.json")

    ex = json.loads((HERE / "example.json").read_text())
    put("example", ex, "poster/example.json (poster/pick_example.py, " + ex["source"] + ")")

    for key, (value, source) in DOC_NUMBERS.items():
        put(key, value, source + " (typed from the card)")
    put("details_url", DETAILS_URL, "poster/build_poster.py")
    return n


def qr_svg(url: str, path: Path) -> None:
    import qrcode
    import qrcode.image.svg

    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    img.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, help="also write a PNG of the whole page here")
    ap.add_argument("--ppi", type=float, default=30.0, help="preview resolution")
    args = ap.parse_args()

    out = HERE / "numbers.json"
    out.write_text(json.dumps(numbers(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    import charts
    import details

    charts.build(ROOT, HERE / "figures")
    qr_svg(DETAILS_URL, HERE / "figures" / "qr.svg")
    print("wrote poster/figures/")
    details.write(json.loads(out.read_text()), HERE / "DETAILS.md")
    print("wrote poster/DETAILS.md")

    import typst

    pdf = HERE / "poster.pdf"
    typst.compile(str(HERE / "poster.typ"), output=str(pdf), root=str(ROOT))
    print(f"wrote {pdf.relative_to(ROOT)}")
    if args.preview:
        typst.compile(str(HERE / "poster.typ"), output=str(args.preview), root=str(ROOT),
                      format="png", ppi=args.ppi)
        print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
