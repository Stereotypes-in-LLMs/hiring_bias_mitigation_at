// GTC 2027 poster, first draft. 44 x 35 in, landscape.
// Every number comes from numbers.json, and cropped figures from figures/, both written by
// build_poster.py from the repository's result tables and figures. Dashed boxes are figures still to be drawn. Text in [square brackets] is
// still to be agreed.

#let N = json("numbers.json")

#let ink = rgb("#2C2C2A")
#let muted = rgb("#5F5E5A")
#let mine = (fg: rgb("#0C447C"), bg: rgb("#E6F1FB"), line: rgb("#185FA5"))
#let marta = (fg: rgb("#3C3489"), bg: rgb("#EEEDFE"), line: rgb("#534AB7"))
#let shared = (fg: rgb("#444441"), bg: rgb("#F1EFE8"), line: rgb("#B4B2A9"))

#set page(width: 44in, height: 35in, margin: 0.9in, fill: white)
#set text(font: "Liberation Sans", size: 25pt, fill: ink)
#set par(leading: 0.5em, spacing: 0.75em)
#set list(spacing: 0.7em, indent: 0.2em, marker: text(fill: muted, sym.bullet))

#let panel(title, owner: mine, fill-height: true, body) = block(
  width: 100%, height: if fill-height { 100% } else { auto }, stroke: 2pt + owner.line, radius: 16pt, clip: true, {
    block(width: 100%, fill: owner.bg, inset: (x: 26pt, y: 18pt),
      text(size: 38pt, weight: "bold", fill: owner.fg, title))
    block(width: 100%, inset: (x: 26pt, top: 20pt, bottom: 20pt), body)
  })

#let sub(body) = block(above: 1.1em, below: 0.6em,
  text(size: 29pt, weight: "bold", fill: mine.fg, body))

#let todo(height, body) = block(
  width: 100%, height: height, radius: 12pt, inset: 22pt,
  stroke: (paint: shared.line, thickness: 3pt, dash: "dashed"),
  align(center + horizon, text(size: 22pt, fill: muted, body)))

#let callout(owner: shared, body) = block(
  width: 100%, fill: owner.bg, radius: 12pt, inset: 20pt, above: 1em, body)

#let cap(body) = block(above: 0.5em, below: 1em, text(size: 19pt, fill: muted, body))

// The audit, drawn: one CV, its variants in one group, one decision each.
#let decision(body, odd: false) = box(
  width: 100%, fill: if odd { rgb("#FBEAE3") } else { white }, radius: 8pt, inset: (y: 8pt),
  stroke: 1.5pt + if odd { rgb("#eb6834") } else { shared.line },
  align(center, text(size: 22pt, weight: "bold", body)))
#let variant(body) = block(
  width: 100%, fill: mine.bg, radius: 8pt, inset: (x: 10pt, y: 10pt), below: 6pt,
  align(center, text(size: 21pt, fill: mine.fg, body)))
#let audit-schematic = {
  set par(leading: 0.4em)
  align(center, block(fill: shared.bg, radius: 10pt, inset: (x: 18pt, y: 12pt),
    text(size: 23pt)[*One job ad + one CV*, from #N.audit_pairs pairs]))
  align(center, block(above: 6pt, below: 6pt, text(size: 20pt, fill: muted)[#sym.arrow.b insert one value of the attribute #sym.arrow.b]))
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 10pt,
    { variant["combat", implicit]; decision[hire] },
    { variant["civilian", implicit]; decision[hire] },
    { variant[other values, implicit]; decision(odd: true)[reject] },
  )
  block(above: 12pt, text(size: 21pt, fill: muted)[One *set* = the variants of one CV in one group and condition. Decisions differ, so this set is *unstable* (illustration). An attribute-free copy of each CV gives the hire rate.])
}

// The route the attribute takes in the base model (Marta's patching).
#let hop(title, detail) = block(
  width: 100%, height: 100%, fill: marta.bg, radius: 10pt, inset: (x: 8pt, y: 8pt),
  align(center + horizon, {
    text(size: 18pt, weight: "bold", fill: marta.fg, title)
    linebreak()
    text(size: 16pt, fill: marta.line, detail)
  }))
#let pathway = grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  rows: 1.45in,
  column-gutter: 4pt,
  align: horizon,
  hop[Attribute][words in the CV],
  text(size: 24pt, fill: marta.line, sym.arrow.r),
  hop[Hub][layer-15 attention, head L15.H11],
  text(size: 24pt, fill: marta.line, sym.arrow.r),
  hop[L#N.pathway_layers][effect reaches the decision position],
  text(size: 24pt, fill: marta.line, sym.arrow.r),
  hop[Readout][MLP 31 → hire or reject],
)

#let stage(title, detail) = block(
  width: 100%, fill: mine.bg, radius: 10pt, inset: (x: 18pt, y: 12pt), below: 0pt, {
    text(weight: "bold", fill: mine.fg, title)
    linebreak()
    text(size: 20pt, fill: mine.line, detail)
  })
#let down = align(center, block(above: 4pt, below: 4pt, text(size: 24pt, fill: shared.line, sym.arrow.b)))

#let hero(number, label) = block(
  width: 100%, height: 100%, fill: shared.bg, radius: 16pt, inset: (x: 28pt, y: 20pt), {
    text(size: 72pt, weight: "bold", fill: ink, number)
    v(0.2em)
    text(size: 24pt, fill: muted, label)
  })

#grid(
  rows: (auto, 3.2in, 1fr, auto),
  gutter: 0.45in,

  // Title band.
  grid(
    columns: (1fr, 7in),
    gutter: 0.6in,
    {
      text(size: 82pt, weight: "bold")[What training after SFT buys for stable LLM hiring decisions, and what it costs]
      v(0.15em)
      text(size: 34pt, fill: muted)[Qwen3.5-9B on an English counterfactual hiring audit, trained and evaluated on NVIDIA GH200 (CSCS Alps)]
      v(0.1em)
      text(size: 30pt)[\[Authors and affiliations, order to be agreed\]]
    },
    grid(columns: 2, gutter: 0.3in,
      todo(2.6in, [logos]),
      todo(2.6in, [QR code \ repository])),
  ),

  // Headline numbers.
  grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.45in,
    hero[#N.distill_mean pp][best mean change in unstable sets over three training seeds (decision distillation, sd #N.distill_sd), at its own hire rate],
    hero[#N.dpo_held][DPO training seeds whose 95% interval stays below 0. Seed 42's #N.dpo_seeds.at(0).own pp did not replicate],
    hero[#N.explore_ratio][GPU-hours spent exploring GRPO and DAPO (#N.explore_rl) against DPO (#N.explore_dpo)],
    hero[#N.b_speedup][audit generation speed from vLLM prefix caching on one GH200, against cache off and shuffled order],
  ),

  // Main row.
  grid(
    columns: (22fr, 34fr, 24fr, 20fr),
    gutter: 0.45in,

    panel[The test][
      #audit-schematic
      #v(0.3em)
      *#N.audit_pairs* job–CV pairs, *#N.audit_prompts* prompts, *#N.audit_sets* within-group sets.

      *Unstable set:* the attribute variants of one CV do not all get the same decision. We report the share of unstable sets.

      #sub[How two models are compared]
      - At an equal hire rate: SFT-v3's #N.common_hire, where it has #N.ref_unstable unstable sets
      - 95% CI over candidates
      - Guardrails: recall of GPT-4o's hires, agreement with GPT-4o, undecided, merge kept
      - Repeated over training seeds; checkpoint declared before the audit

      #callout[*Why the hire rate must match.* Our GRPO pilot looked more stable, #N.grpo_pilot_delta pp, only because it hired less: #N.grpo_pilot_hire. Every decision it changed went from hire to reject.]

      #callout(owner: mine)[*Starting point.* SFT-v3, trained on 8-vote anchors: #N.sft_v3_vs_v1 pp against the first SFT and #N.sft_v3_vs_base pp against the base model (one seed). Every recipe here starts from it.]
    ],

    panel[A. What training after SFT buys, and what it costs][
      #image("figures/cost_quality.svg", width: 100%)
      #cap[Every arm moved to SFT-v3's hire rate. Below 0 is more stable than SFT-v3. From `reports/final/final_table.csv` and `recipes.csv`.]

      #table(
        columns: (2.6fr, 0.8fr, 1.6fr, 1.6fr, 1.1fr),
        stroke: none,
        inset: (x: 10pt, y: 9pt),
        fill: (_, y) => if y == 0 { shared.bg } else if calc.even(y) { rgb("#FAFAF8") },
        align: (x, _) => if x == 0 { left } else { right },
        table.header(
          [*Recipe after SFT-v3*], [*Seeds*], [*Δ own hire rate, pp (sd)*], [*Δ at #N.common_hire, pp (sd)*], [*GPU-h*]),
        ..N.recipes.map(r => (
          if r.note != "" [#r.label #text(size: 19pt, fill: muted)[(#r.note)]] else [#r.label],
          r.seeds, r.own, r.common, r.gpu_h)).flatten(),
      )
      #cap[Means over training seeds, `reports/final/recipes.csv`. Every recipe passes every guardrail. GPU-h: data and training for one model.]

      #grid(
        columns: (1.75fr, 1fr),
        gutter: 0.35in,
        image("figures/curves.svg", width: 100%),
        {
          cap[The same comparison at every hire rate from 10% to 35%. The seed mean is below SFT-v3 at #N.curve_below.dpo hire rates for DPO, #N.curve_below.distill for distillation and #N.curve_below.grpo for GRPO + decay. From `reports/final/curves.csv`.]
          callout(owner: mine)[*One seed is not a result.* DPO seed 42: #N.dpo_seeds.at(0).own pp #N.dpo_seeds.at(0).ci. Seeds 43 and 44, same data and settings: #N.dpo_seeds.at(1).own and #N.dpo_seeds.at(2).own pp.]
        },
      )
    ],

    grid(
      rows: (auto, 1fr),
      gutter: 0.45in,
      panel(owner: marta, fill-height: false)[D. Where the attribute reaches the decision][
        #image("figures/residual.svg", width: 100%)
        #cap[Patching the residual stream at layer L with the other variant's removes this much of the attribute effect. The dashed rise is the effect reaching the last tokens, where the decision is read. Grey bands: just after a full-attention layer. Combat vs civilian, implicit; the same 64 pairs for both models. From `reports/interp/patching/{base,sft}-grid-mil_combat_civ_impl-n64-s42/real.csv`.]

        In the base model the attribute reaches the decision at layers *#N.pathway_layers*, through a hub in layer-15 attention (head *L15.H11*), and is read out by *MLP 31*.

        SFT-v3 removes *#N.sft_v3_effect_removed* of the per-pair effect. What is left takes the same route, and L15.H11 is still its largest head.

        #pathway
      ],
      panel[D. What later training does to that pathway][
        - *Distillation* keeps the pathway and shrinks it: #N.distill_kept of SFT-v3's attribute effect is left (seed 43), and L15.H11 is still the top head.
        - *DPO and GRPO* widen every margin, #N.dpo_scale and #N.grpo_scale. GRPO keeps SFT-v3's pathway; DPO's shifts toward later layers.

        #callout[*Control.* Rescaling SFT-v3's margins by any factor changes the matched metric by #N.scale_change. The gains sit on borderline pairs instead: #N.borderline_fixed_share of the sets distillation fixes are on pairs near the cut (post hoc).]
      ],
    ),

    panel[B + C. The system on GH200][
      #stage[Fixed data split][#N.holdout_candidates held-out candidates, leakage check]
      #down
      #stage[Sampling and teaching][vLLM per GH200; 122B teacher, tensor parallel 4]
      #down
      #stage[Post-training][TRL and PEFT, LoRA r32, 4 GH200 (DDP)]
      #down
      #stage[Merge adapters][SFT, then the later stage; at least 80% of the update kept]
      #down
      #stage[Batch audit][vLLM on 1 GH200, greedy, #N.audit_prompts prompts]
      #down
      #stage[HF check and patching][decisions agree on #N.hf_vllm_agreement]
      #down
      #stage[Metrics and cost ledger][every GPU step metered]

      #sub[B. The audit as an inference load]
      Against cache off and shuffled order:
      - as the audits run: *#N.b_saved* less generation time (#N.b_speedup), cache hit rate #N.b_hit_built
      - sorted by prefix: a further #N.b_order less, hit rate #N.b_hit_prefix

      #N.b_token_shared of prompt tokens are shared token for token, but vLLM 0.31 reuses only whole blocks of #N.b_block tokens on this hybrid model: a bound of #N.b_block_bound.

      #sub[C. Serving is part of the measurement]
      A null result over #N.lora_null_adapters came from vLLM's LoRA path, which does not reproduce Qwen3.5 adapters. We serve merged weights and check them against HF (DPO's merge kept #N.dpo_merge_kept).

      #text(fill: muted)[Every GPU step is metered: #N.gpu_h_total GPU-h allocated for the whole project since October; one evaluation uses #N.eval_gpu_h GPU-h.]
    ],
  ),

  // Footer.
  block(width: 100%, fill: shared.bg, radius: 16pt, inset: (x: 28pt, y: 22pt),
    grid(
      columns: (2.1fr, 1.3fr, 1.2fr),
      gutter: 0.6in,
      {
        block(below: 0.5em, text(size: 30pt, weight: "bold")[Conclusions])
        set text(size: 23pt)
        enum(
          [At an equal hire rate and over three seeds, no recipe after SFT is clearly more stable than SFT-v3; single-seed wins shrank or did not replicate.],
          [The cheap recipes match the expensive ones: exploring GRPO and DAPO cost #N.explore_ratio the GPU-hours of DPO.],
          [None removes the attribute pathway. They rescale margins, and rescaling alone does not change the metric.],
        )
      },
      {
        block(below: 0.5em, text(size: 30pt, weight: "bold")[Limitations])
        set text(size: 23pt)
        [English only; one 9B model; #N.audit_candidates benchmark candidates. Patching on 64 pairs per contrast, one selection. GRPO checkpoint 525 was picked with audit numbers in view.]
      },
      {
        block(below: 0.5em, text(size: 30pt, weight: "bold")[\[Contributions, to be agreed\]])
        set text(size: 23pt)
        [Post-training, evaluation standard, system and cost; activation-patching pipeline (M. Samoilenko); benchmark and audit pipeline (N. Drushchak et al.). Compute: CSCS Alps. Every number is in `poster/numbers.json` with its source file.]
      },
    )),
)
