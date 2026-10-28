"""Decision distillation against SFT-v3: the three measures registered in docs/DISTILL_PATCHING.md.

    python scripts/interp/20_distill_compare.py --spec distill43 --ref sft_v3 \\
        --contrast mil_combat_civ_impl --contrast mil_combat_civ_expl \\
        --contrast rel_atheist_christian_impl

    (a) sensitivity  rho = r_attr / r_scale per contrast, on every aligned pair: r_attr is mean
                     |dm| of spec over ref, r_scale median |m| on the attribute-free prompts of
                     the same candidates, spec over ref; 95% CI by a bootstrap over candidates
    (b) components   real-patching contribution per component (suffix positions, logits per
                     pair) on the identical patched pairs, spec over ref ("kept"); AtP for heads
    (c) residual     per depth (the residual entering each layer, 0 = embeddings, 32 = the
                     final pre-norm residual), mean over aligned pairs of
                     |r_L(A) - r_L(B)| / |r_L(B)| at the last position (02_margins.py --resid),
                     spec over ref

Inputs come from outputs/interp/{margins,manifest} and reports/interp/patching/; outputs go to
reports/interp/distill/<spec>/ (CSVs and residual.svg). No GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.interp.io import interp_dir, report_dir  # noqa: E402

HUB = (("mixer", 11), ("mixer", 15))
READOUT = (("mlp", 31),)
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


def margins(spec: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(interp_dir("margins", spec) / f"{name}.parquet")


def sensitivity(manifest, m_ref, m_spec, spec, ref, contrasts, n_boot, seed) -> pd.DataFrame:
    free = m_ref[m_ref["condition"] == "attr_free"][["candidate_id", "prompt_uid", "margin_fp32"]]
    free = free.merge(m_spec[["prompt_uid", "margin_fp32"]], on="prompt_uid",
                      suffixes=("_ref", "_spec"))
    rng = np.random.default_rng(seed)
    out = []
    for c in contrasts:
        rows = manifest[(manifest["contrast_id"] == c) & manifest["aligned"]]
        cands = np.array(sorted(set(rows["candidate_id"]) & set(free["candidate_id"])))
        per = rows.assign(a_ref=rows[f"dm_{ref}"].abs(), a_spec=rows[f"dm_{spec}"].abs())
        per = per.groupby("candidate_id")[["a_ref", "a_spec"]].agg(["sum", "count"])
        fr = free.assign(f_ref=free["margin_fp32_ref"].abs(), f_spec=free["margin_fp32_spec"].abs())
        fr = fr.set_index("candidate_id")

        def rho(chosen, per=per, fr=fr):
            p = per.loc[chosen]
            r_attr = p[("a_spec", "sum")].sum() / p[("a_ref", "sum")].sum()
            f = fr.loc[chosen]
            r_scale = np.median(f["f_spec"]) / np.median(f["f_ref"])
            return r_attr, r_scale, r_attr / r_scale

        r_attr, r_scale, value = rho(cands)
        boots = np.array([rho(cands[rng.integers(0, len(cands), len(cands))])[2]
                          for _ in range(n_boot)])
        out.append({"contrast": c, "pairs": len(rows), "candidates": len(cands),
                    f"mean_abs_dm_{ref}": float(per[("a_ref", "sum")].sum() / len(rows)),
                    f"mean_abs_dm_{spec}": float(per[("a_spec", "sum")].sum() / len(rows)),
                    "r_attr": r_attr, "r_scale": r_scale, "rho": value,
                    "rho_ci_low": float(np.quantile(boots, 0.025)),
                    "rho_ci_high": float(np.quantile(boots, 0.975))})
    return pd.DataFrame(out)


def components(spec, ref, contrasts, suffix) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, heads = [], []
    for c in contrasts:
        d = {s: report_dir("patching", f"{s}-grid-{c}{suffix}") for s in (ref, spec)}
        if not all((d[s] / "real.csv").exists() for s in d):
            continue
        real = {s: pd.read_csv(d[s] / "real.csv") for s in d}
        pairs = {s: pd.read_csv(d[s] / "pairs.csv") for s in d}
        if set(pairs[ref]["pair_uid"]) != set(pairs[spec]["pair_uid"]):
            raise SystemExit(f"{c}: the two specs were patched on different pairs")
        total = {s: float((pairs[s]["m_a"] - pairs[s]["m_b"]).abs().mean()) for s in d}
        key = ["kind", "layer", "cls"]
        j = real[ref].merge(real[spec], on=key, suffixes=("_ref", "_spec"))
        j = j[j["kind"].isin(["mixer", "mlp"]) & (j["cls"] == "suffix")]
        j = j.assign(contrast=c, kept=j["removed_logits_per_pair_spec"]
                     / j["removed_logits_per_pair_ref"],
                     total_ref=total[ref], total_spec=total[spec],
                     total_kept=total[spec] / total[ref])
        rows.append(j)
        atp = {s: pd.read_csv(d[s] / "atp.csv") for s in d}
        h = atp[ref].merge(atp[spec], on=["kind", "layer", "head", "cls"],
                           suffixes=("_ref", "_spec"))
        h = h[(h["kind"] == "head_in") & (h["cls"] == "suffix")]
        h = h.assign(contrast=c, kept=h["removed_logits_per_pair_spec"]
                     / h["removed_logits_per_pair_ref"])
        heads.append(h.sort_values("removed_logits_per_pair_ref", ascending=False).head(12))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows, ignore_index=True), pd.concat(heads, ignore_index=True)


def resid_rows(spec: str, name: str, uids: set[str]) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    out = {}
    for part in sorted(interp_dir("margins", spec, name).glob("part-*.parquet")):
        meta = pd.read_parquet(part, columns=["prompt_uid"])
        keep = np.flatnonzero(meta["prompt_uid"].isin(uids).to_numpy())
        if not len(keep):
            continue
        resid = load_file(str(part.with_name(part.name.replace("part-", "resid-")
                                             .replace(".parquet", ".safetensors"))))["resid"]
        for i in keep:
            out[meta["prompt_uid"].iat[i]] = resid[i].astype(np.float32)
    return out


def residual(manifest, spec, ref, name, contrasts) -> pd.DataFrame:
    rows = manifest[manifest["contrast_id"].isin(contrasts) & manifest["aligned"]]
    uids = set(rows["uid_a"]) | set(rows["uid_b"])
    r = {s: resid_rows(s, name, uids) for s in (ref, spec)}
    out = []
    for c, g in rows.groupby("contrast_id"):
        rel = {}
        for s in (ref, spec):
            a = np.stack([r[s][u] for u in g["uid_a"]])
            b = np.stack([r[s][u] for u in g["uid_b"]])
            rel[s] = (np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)).mean(0)
        for layer in range(len(rel[ref])):
            out.append({"contrast": c, "depth": layer, f"rel_diff_{ref}": rel[ref][layer],
                        f"rel_diff_{spec}": rel[spec][layer],
                        "ratio": rel[spec][layer] / rel[ref][layer]})
    return pd.DataFrame(out)


def svg(table: pd.DataFrame, path: Path, title: str) -> None:
    w, h, left, top = 720, 320, 60, 40
    depth = int(table["depth"].max())
    ymax = max(1.4, float(table["ratio"].max()) * 1.05)

    def X(v):
        return left + v / depth * (w - left - 200)

    def Y(v):
        return top + (h - top - 40) * (1 - v / ymax)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           'font-family="sans-serif" font-size="11">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{left}" y="18" font-weight="bold">{title}</text>',
           f'<rect x="{X(15):.1f}" y="{top}" width="{X(20) - X(15):.1f}" '
           f'height="{h - top - 40}" fill="#f2f2f2"/>']
    for t in (0, 0.5, 0.85, 1.0, 1.15):
        out.append(f'<line x1="{left}" x2="{X(depth):.1f}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                   f'stroke="{"#999" if t == 1 else "#e0e0e0"}"/>')
        out.append(f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for s in range(0, depth + 1, 4):
        out.append(f'<text x="{X(s):.1f}" y="{h - 24}" text-anchor="middle">{s}</text>')
    for i, (c, g) in enumerate(table.groupby("contrast", sort=False)):
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g["depth"], g["ratio"]))
        color = COLORS[i % len(COLORS)]
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        out.append(f'<text x="{X(depth) + 8:.1f}" y="{top + 14 * i}" fill="{color}">{c}</text>')
    out.append(f'<text x="{w / 2 - 80}" y="{h - 6}" text-anchor="middle">residual depth '
               '(the residual entering each layer; shaded 15-20)</text></svg>')
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ref", default="sft_v3")
    ap.add_argument("--contrast", action="append", required=True)
    ap.add_argument("--patch-suffix", default="-n64-s42-sel_sft_v3")
    ap.add_argument("--name", default="grid-en")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = report_dir("distill", args.spec)
    manifest = pd.read_parquet(interp_dir("manifest") / f"contrasts-{args.name}.parquet")
    pd.set_option("display.width", 200)

    sens = sensitivity(manifest, margins(args.ref, args.name), margins(args.spec, args.name),
                       args.spec, args.ref, args.contrast, args.n_boot, args.seed)
    sens.to_csv(out / "sensitivity.csv", index=False)
    print("(a) sensitivity\n" + sens.round(3).to_string(index=False))

    comp, heads = components(args.spec, args.ref, args.contrast, args.patch_suffix)
    if len(comp):
        comp.to_csv(out / "components.csv", index=False)
        heads.to_csv(out / "heads.csv", index=False)
        cols = ["contrast", "kind", "layer", "removed_logits_per_pair_ref",
                "removed_logits_per_pair_spec", "kept", "total_kept"]
        big = comp[comp["removed_logits_per_pair_ref"] >= 0.1]
        print("\n(b) components contributing >= 0.1 logits/pair under the reference\n"
              + big[cols].round(3).to_string(index=False))
        cols = ["contrast", "layer", "head", "removed_logits_per_pair_ref",
                "removed_logits_per_pair_spec", "kept"]
        print("\n(b) top heads by AtP under the reference\n"
              + heads.groupby("contrast").head(5)[cols].round(3).to_string(index=False))
        for c, g in comp.groupby("contrast"):
            idx = g.set_index(["kind", "layer"])
            hub = sum(idx.loc[k, "removed_logits_per_pair_spec"] for k in HUB) / sum(
                idx.loc[k, "removed_logits_per_pair_ref"] for k in HUB)
            ro = sum(idx.loc[k, "removed_logits_per_pair_spec"] for k in READOUT) / sum(
                idx.loc[k, "removed_logits_per_pair_ref"] for k in READOUT)
            print(f"{c}: hub kept {hub:.3f}, readout kept {ro:.3f}, difference {hub - ro:+.3f}, "
                  f"total effect kept {g['total_kept'].iat[0]:.3f}")
    else:
        print("\n(b) no patching summaries for both specs yet")

    if any(interp_dir("margins", args.spec, args.name).glob("resid-*.safetensors")):
        res = residual(manifest, args.spec, args.ref, args.name, args.contrast)
        res.to_csv(out / "residual.csv", index=False)
        svg(res, out / "residual.svg",
            f"(c) attribute difference in the last-position residual, {args.spec} / {args.ref}")
        mid = res[res["depth"].between(15, 20)].groupby("contrast")["ratio"].mean()
        last = res[res["depth"] == res["depth"].max()].set_index("contrast")["ratio"]
        print("\n(c) residual ratio, mean over depths 15-20 and at the final depth\n"
              + pd.DataFrame({"depth_15_20": mid, "final": last}).round(3).to_string())


if __name__ == "__main__":
    main()
