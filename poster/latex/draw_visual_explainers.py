"""Draw editable poster explainers from the existing measured examples.

These are standalone figure candidates. This script does not edit main.tex.
No inference, selection of new training settings, or external images are used.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, Circle

ROOT = Path(__file__).resolve().parents[2]
INK, GREEN, PALE = "#18384A", "#426F20", "#E4F3CE"
GRAY, MIST, RULE, PANEL = "#83929B", "#B8CAD4", "#D5DEDF", "#F3F5F4"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})


def canvas(h=940):
    fig = plt.figure(figsize=(18, h / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1800), ylim=(h, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=18, weight="normal", color=INK, ha="left", **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va="top", linespacing=1.24, **kw)


def box(ax, x, y, w, h, fill="white", edge=RULE, lw=1.5):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=lw))


def line(ax, pts, color=GRAY, lw=2, **kw):
    ax.plot(*zip(*pts), color=color, lw=lw, **kw)


def arrow(ax, start, end, color=GRAY, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                mutation_scale=17, color=color, lw=lw))


def step(ax, x, y, n, label):
    ax.add_patch(Circle((x + 17, y + 17), 17, color=INK))
    text(ax, x + 17, y + 3, str(n), 16, "bold", "white", "center")
    text(ax, x + 46, y + 2, label, 21, "bold")


def document_icon(ax, x, y, w=48, h=60):
    ax.add_patch(Polygon([(x, y), (x+w-13, y), (x+w, y+13),
                          (x+w, y+h), (x, y+h)], facecolor="white",
                         edgecolor=INK, lw=1.7))
    line(ax, [(x+w-13, y), (x+w-13, y+13), (x+w, y+13)], INK, 1.3)
    for yy, ww in ((25, 25), (35, 25), (45, 18)):
        line(ax, [(x+10, y+yy), (x+10+ww, y+yy)], MIST, 2)


def pill(ax, x, y, label, hire=True, w=220):
    box(ax, x, y, w, 76, PALE if hire else "white", GREEN if hire else INK, 1.8)
    text(ax, x+w/2, y+17, label, 25, "bold", GREEN if hire else INK, "center")


def save(fig, out, stem):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180, facecolor="white")
        if ext == "svg":
            path = out / f"{stem}.{ext}"
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def measured_case(out, tex):
    values = {}
    for key in ("CaseYears", "CaseBaseRejects", "CaseVariants", "CaseJob"):
        values[key] = re.search(r"\\newcommand\{\\" + key + r"\}\{([^}]*)\}", tex).group(1)
    assert values == {"CaseYears": "4.5", "CaseBaseRejects": "5", "CaseVariants": "20", "CaseJob": "Sales Manager"}
    fig, ax = canvas()
    text(ax, 50, 35, "One CV. One job. Different identity cues.", 30, "bold")
    text(ax, 50, 96, "A measured example of identity-cue instability", 18, color=GREEN)
    step(ax, 50, 172, 1, "Keep qualifications fixed")
    step(ax, 555, 172, 2, "Change the identity cue")
    step(ax, 1065, 172, 3, "Compare decisions")

    for y, name, detail in ((270, "CV", "4.5 years\nin sales roles"),
                             (465, "Sales Manager", "Requires 3+ years\nof sales experience")):
        box(ax, 50, y, 365, 150, PANEL)
        document_icon(ax, 70, y+25)
        text(ax, 140, y+22, name, 20, "bold")
        text(ax, 140, y+65, detail, 17)
    line(ax, [(432,345),(470,345),(470,540),(432,540)], INK, 1.8)
    arrow(ax, (470,442), (510,442), INK)
    line(ax, [(510,345),(510,540)], INK, 1.8)
    arrow(ax, (510,345),(555,345), INK)
    arrow(ax, (510,540),(555,540), INK)

    for y, identity, pronouns in ((270, "female", "she / her"), (465, "male", "he / him")):
        box(ax, 555, y, 395, 150)
        text(ax, 577, y+20, "Same CV + same job", 16, color=GRAY)
        box(ax, 577, y+60, 350, 63, PALE, PALE)
        text(ax, 595, y+72, identity, 22, "bold", GREEN)
        text(ax, 770, y+77, pronouns, 18, color=GREEN)
        arrow(ax, (966,y+75),(1025,y+75), INK)

    text(ax, 1180, 235, "Base", 23, "bold", ha="center")
    text(ax, 1540, 235, "SFT-v3", 23, "bold", ha="center")
    line(ax, [(1360,260),(1360,632)], RULE, 1.5)
    pill(ax, 1070, 307, "Hire")
    pill(ax, 1070, 502, "Reject", False)
    pill(ax, 1430, 307, "Hire")
    pill(ax, 1430, 502, "Hire")
    text(ax, 1180, 600, "Different decisions", 18, "bold", ha="center")
    text(ax, 1540, 600, "Same decision", 18, "bold", GREEN, "center")
    text(ax, 555, 636, "Two of the 20 gender-cue variants shown", 15, color=GRAY)

    box(ax, 50, 702, 1650, 155, PANEL, PANEL)
    text(ax, 72, 720, "All 20 variants", 20, "bold")
    text(ax, 72, 768, "Same CV-job pair", 16)
    for y, label, nhire in ((727, "Base", 15), (790, "SFT-v3", 20)):
        text(ax, 435, y+3, label, 18, "bold")
        for i in range(20):
            box(ax, 585+i*38, y, 28, 28, GREEN if i < nhire else "white",
                GREEN if i < nhire else INK, 1.2)
        text(ax, 1390, y+1, f"{nhire} hire / {20-nhire} reject", 19, "bold")
    text(ax, 50, 883, "Greedy decoding; one illustrative pair. Tiles show counts, not cue order. Consistency does not establish decision correctness.", 13)
    save(fig, out, "same-cv-audit")
    return values


def read_csv(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def decision_geometry(out, source):
    rows = [r for r in read_csv(source/"common_rate_pair_features.csv")
            if (r["backend"],r["recipe"],r["seed"],r["contrast"]) ==
            ("vllm","DPO","42","mil_combat_civ_impl")]
    cuts = {r["rate_label"]:float(r["threshold"]) for r in read_csv(source/"cuts.csv")
            if (r["backend"],r["recipe"],r["seed"]) == ("vllm","DPO","42")}
    t35, t22 = cuts["35"], cuts["sft_common"]
    # Deterministic illustrative selection within one checkpoint and contrast.
    large = max((r for r in rows if r["top1pct_including_ties"] == "True"
                  and float(r["crossing_weight"]) == 1
                  and min(float(r["a"]),float(r["b"])) > t35),
                 key=lambda r:float(r["absolute_effect"]))
    small = max((r for r in rows if min(float(r["a"]),float(r["b"])) < t35 <
                 max(float(r["a"]),float(r["b"]))), key=lambda r:float(r["absolute_effect"]))
    assert large["pair_hash"] == "216583fef5661a4c779b"
    assert small["pair_hash"] == "bb4540e0134d7fdf6141"
    assert min(float(small["a"]),float(small["b"])) < t35 < max(float(small["a"]),float(small["b"])) < t22

    fig, ax = canvas(1080)
    text(ax, 50, 34, "A large logit effect need not flip a decision", 29, "bold")
    text(ax, 50, 94, "Two measured pairs; decisions at two calibrated hire rates", 18, color=GREEN)
    text(ax, 50, 145, "DPO seed 42  |  combat / civilian, implicit cue  |  served vLLM scores", 15)

    xmin, xmax, x0, x1 = -40, 30, 380, 1215
    xx = lambda v: x0+(float(v)-xmin)/(xmax-xmin)*(x1-x0)
    y0, y1 = 278, 558
    box(ax, xx(t35), y0, x1-xx(t35), y1-y0, PALE, PALE)
    text(ax, xx(t35)+14, 293, "Hire at the 35% operating point", 14, color=GREEN)
    for t, label, color, yy in ((t35,"35% hire", GREEN, 211), (t22,"22.51% hire", INK, 211)):
        line(ax, [(xx(t),250),(xx(t),y1+16)], color, 1.8, linestyle="--")
        text(ax, xx(t), yy, label, 16, "bold", color, "center")
        text(ax, xx(t), yy+29, f"cut {t:.2f}", 13, color=color, ha="center")

    text(ax, 50, 335, "Large logit effect", 21, "bold")
    text(ax, 50, 380, "32.06 logits", 25, "bold", GREEN)
    text(ax, 50, 427, "Top 1% in this contrast", 14)
    text(ax, 50, 485, "Smaller logit effect", 21, "bold")
    text(ax, 50, 530, "1.56 logits", 25, "bold", GREEN)
    text(ax, 50, 577, "Near the 35% decision cut", 14)
    for r, y in ((large,390),(small,520)):
        aa, bb = float(r["a"]), float(r["b"])
        line(ax, [(xx(aa),y),(xx(bb),y)], INK, 3)
        ax.plot(xx(aa),y,"o",ms=10,mec=INK,mfc=INK,mew=1.7)
        ax.plot(xx(bb),y,"o",ms=10,mec=INK,mfc="white",mew=1.7)
        if r is large:
            text(ax,xx(aa),y+22,f"{aa:.2f}",14,ha="center")
            text(ax,xx(bb),y+22,f"{bb:.2f}",14,ha="center")

    line(ax, [(x0,606),(x1,606)], GRAY, 1.4)
    for tick in (-40,-20,0,20,30):
        line(ax,[(xx(tick),601),(xx(tick),612)],GRAY,1.3)
        text(ax,xx(tick),619,str(tick),13,ha="center")
    text(ax,(x0+x1)/2,653,"Decision margin (logits); higher favors Hire",16,ha="center")

    text(ax, 1340, 255, "22.51% hire", 17, "bold", ha="center")
    text(ax, 1610, 255, "35% hire", 17, "bold", ha="center")
    for y, labels in ((354,("Split", "Both hire")), (484,("Both reject", "Split"))):
        for x,label in zip((1240,1510),labels):
            box(ax,x,y,200,74,PALE if label=="Both hire" else "white", GREEN if label=="Both hire" else INK)
            text(ax,x+100,y+21,label,18,"bold",GREEN if label=="Both hire" else INK,"center")

    # Explicitly labeled zoom. The main rows above share an unbroken x scale.
    box(ax, 50, 716, 1660, 228, PANEL, PANEL)
    text(ax, 75, 736, "Zoom at the 35% cut", 20, "bold")
    zz=lambda v: 500+(float(v)+34.25)/2.9*660
    cutx=zz(t35)
    box(ax,cutx,794,1160-cutx,87,PALE,PALE)
    line(ax,[(cutx,783),(cutx,882)],GREEN,1.8,linestyle="--")
    text(ax,cutx,752,"35% cut",14,"bold",GREEN,"center")
    aa, bb=float(small["a"]),float(small["b"])
    line(ax,[(zz(aa),832),(zz(bb),832)],INK,3)
    ax.plot(zz(aa),832,"o",ms=11,mec=INK,mfc=INK,mew=1.8)
    ax.plot(zz(bb),832,"o",ms=11,mec=INK,mfc="white",mew=1.8)
    text(ax,zz(bb),858,f"{bb:.2f}\nReject",15,ha="center")
    text(ax,zz(aa),858,f"{aa:.2f}\nHire",15,ha="center")
    text(ax,75,803,"Same two identities.\nDifferent sides of the threshold.",17)
    text(ax,1240,789,"Distance to the cut\nmatters alongside\nlogit-effect size.",21,"bold")
    ax.plot(80,979,"o",ms=8,mec=INK,mfc=INK,mew=1.4)
    text(ax,102,965,"Combat cue",14)
    ax.plot(328,979,"o",ms=8,mec=INK,mfc="white",mew=1.4)
    text(ax,350,965,"Civilian cue",14)
    text(ax,50,1010,"Selected measured cases illustrate threshold geometry; they do not estimate prevalence or explain the whole instability curve.\nEach pair holds its CV and job fixed. Thresholds are calibrated within this same execution path; no endpoint shown is tied at a cut.",13)
    save(fig,out,"effect-versus-decision")
    return {"backend":"vllm", "recipe":"DPO", "seed":42,
            "contrast":"mil_combat_civ_impl", "cut_35":t35, "cut_22_512":t22,
            "large":large,"small":small,
            "selection":"Largest top-1%-effect pair crossing common cut but entirely above 35% cut; largest-effect pair strictly crossing 35% cut. One seed and contrast only."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"reports/research_followup/decision_geometry")
    p.add_argument("--out",type=Path,default=ROOT/"poster/latex/figures/explainers")
    p.add_argument("--case-source",type=Path,default=ROOT/"poster/latex/main.tex",
                   help="Optional read-only snapshot of the latest poster source")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    texpath=a.case_source
    meta={"case":measured_case(a.out,texpath.read_text()),
          "geometry":decision_geometry(a.out,a.source)}
    meta["sources"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (a.source/"common_rate_pair_features.csv",a.source/"cuts.csv")}
    meta["sources"]["poster/latex/main.tex"]=hashlib.sha256(texpath.read_bytes()).hexdigest()
    (a.out/"measured-examples.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":
    main()
