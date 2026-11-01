// GTC 2027 poster. 44 x 35 in, landscape. Six sections; the poster keeps the core and the
// QR code leads to DETAILS.md. Every number comes from numbers.json, written by
// build_poster.py from the repository's result tables. Text in [square brackets] is still to
// be agreed.

#let N = json("numbers.json")
#let ex = N.example

#let ink = rgb("#2C2C2A")
#let muted = rgb("#5F5E5A")
#let mine = (fg: rgb("#0C447C"), bg: rgb("#E6F1FB"), line: rgb("#185FA5"))
#let marta = (fg: rgb("#3C3489"), bg: rgb("#EEEDFE"), line: rgb("#534AB7"))
#let shared = (fg: rgb("#444441"), bg: rgb("#F1EFE8"), line: rgb("#B4B2A9"))
#let reject-fill = rgb("#FBEAE3")
#let reject-line = rgb("#eb6834")

#set page(width: 44in, height: 35in, margin: 0.9in, fill: white)
#set text(font: "Liberation Sans", size: 26pt, fill: ink)
#set par(leading: 0.5em, spacing: 0.8em)
#set list(spacing: 0.6em, indent: 0.2em, marker: text(fill: muted, sym.bullet))

// A numbered section.
#let section(num, title, owner: mine, body) = block(
  width: 100%, height: 100%, stroke: 2pt + owner.line, radius: 16pt, clip: true, {
    block(width: 100%, fill: owner.bg, inset: (x: 26pt, y: 16pt), {
      box(width: 46pt, height: 46pt, radius: 23pt, fill: owner.line,
        align(center + horizon, text(size: 28pt, weight: "bold", fill: white, num)))
      h(14pt)
      text(size: 40pt, weight: "bold", fill: owner.fg, title)
    })
    block(width: 100%, inset: (x: 26pt, top: 18pt, bottom: 18pt), body)
  })

#let note(body) = text(size: 21pt, fill: muted, body)
#let callout(owner: shared, body) = block(
  width: 100%, fill: owner.bg, radius: 12pt, inset: 18pt, above: 0.9em, body)

// ---------------------------------------------------------------- methodology diagram
#let node(title, detail, owner: mine) = block(
  width: 100%, height: 100%, fill: owner.bg, stroke: 1.5pt + owner.line, radius: 10pt,
  inset: (x: 12pt, y: 10pt),
  align(center + horizon, {
    text(size: 25pt, weight: "bold", fill: owner.fg, title)
    linebreak()
    text(size: 21pt, fill: owner.fg, detail)
  }))
#let arrow(owner: mine) = align(center + horizon, text(size: 34pt, fill: owner.line, sym.arrow.r))
#let down(owner: marta) = align(center + horizon, text(size: 34pt, fill: owner.line, sym.arrow.b))
#let stack(..items) = grid(rows: (1fr,) * items.pos().len(), gutter: 8pt, ..items)

#let pipeline = grid(
  columns: (1fr, 0.45in, 1fr, 0.45in, 1fr, 0.45in, 1.1fr, 0.45in, 1fr, 0.45in, 1fr, 0.45in, 1.1fr),
  rows: (4.3in, 0.6in, 1.9in, auto),
  align: horizon,

  node[Training pool][job–CV pairs; no benchmark candidate],
  arrow(),
  node[Teacher votes][#N.teacher, #N.vote_draws draws per prompt],
  arrow(),
  node[SFT-v3][LoRA #N.lora on the voted decisions],
  arrow(),
  stack(node[DPO][on-policy pairs], node[GRPO / DAPO][verifiable reward],
        node[Distillation][own attribute-free decision]),
  arrow(),
  node[Merge][at least #N.min_merge_kept_pct% of the update kept],
  arrow(),
  node[Batch audit][vLLM on GH200, prefix cache, #N.audit_prompts prompts],
  arrow(),
  stack(node[Matched comparison][same hire rate, 95% CI, guardrails],
        node[Cost ledger][GPU-hours of every step]),

  [], [], [], [], [], [], down(), [], [], [], [], [], [],

  grid.cell(colspan: 5, block(width: 100%, height: 100%, inset: (x: 6pt),
    align(left + horizon, text(size: 22pt, fill: marta.fg)[*Activation patching* (Marta Samoilenko): where does the attribute reach the decision?]))),
  [],
  node(owner: marta)[Models][base, SFT-v3, best arms],
  arrow(owner: marta),
  node(owner: marta)[Exact margins][HF, at least #N.patch_gate agreement with the audit],
  arrow(owner: marta),
  node(owner: marta)[Contrast pairs][same CV, two attribute values],
  arrow(owner: marta),
  node(owner: marta)[Patching → pathway][residual, components, heads],

  grid.cell(colspan: 13, block(above: 14pt, note[All on CSCS Alps, NVIDIA GH200 nodes: TRL and PEFT for training, vLLM #N.vllm_version for audits; every GPU step is metered.])),
)

// ---------------------------------------------------------------- dataset example
#let chip(body, odd: false) = box(
  width: 100%, fill: if odd { reject-fill } else { white }, radius: 8pt, inset: (y: 7pt),
  stroke: 1.5pt + if odd { reject-line } else { shared.line },
  align(center, text(size: 22pt, weight: "bold", body)))

#let example-card = block(width: 100%, fill: shared.bg, radius: 12pt, inset: 20pt, {
  set par(leading: 0.45em)
  text(size: 27pt, weight: "bold")[Example: #ex.job_position]
  v(0.3em)
  text(size: 22pt)[*Job:* #ex.job_musts.map(m => ["#m"]).join([ · ])]
  v(0.2em)
  text(size: 22pt)[*CV:* #ex.cv_role. #ex.cv_lines.map(l => ["#l"]).join([ ])]
  v(0.4em)
  text(size: 22pt)[*One sentence added* (implicit wording), and SFT-v3's decision:]
  v(0.2em)
  grid(
    columns: (1fr, 2.3in),
    gutter: 8pt,
    align: horizon,
    ..ex.variants.map(v => (
      text(size: 21pt)["#v.sentence"],
      chip(odd: v.decision != "hire")[#v.decision])).flatten(),
  )
  v(0.4em)
  let vet = ex.variants.find(v => v.value == "War veteran")
  let civ = ex.variants.find(v => v.value == "Civilian")
  text(size: 21pt)[*Reason, war veteran:* "#vet.feedback"]
  v(0.15em)
  text(size: 21pt)[*Reason, civilian:* "#civ.feedback"]
  v(0.3em)
  note[GPT-4o and SFT-v3 on the attribute-free copy: #ex.attr_free_decision. Quoted as written.]
})

// ---------------------------------------------------------------- results table
#let results-table = table(
  columns: (2.3fr, 1fr, 1.2fr, 1.4fr, 1.8fr, 1.1fr),
  stroke: none,
  inset: (x: 10pt, y: 13pt),
  fill: (_, y) => if y == 0 { shared.bg } else if y == 2 { mine.bg },
  align: (x, _) => if x == 0 { left } else { right },
  table.header([*Model*], [*Hire rate*], [*Unstable sets*], [*vs base, pp*],
               [*95% CI or seed range*], [*GPU-h*]),
  ..N.vs_base.map(r => (
    if r.seeds != "" and r.model != "SFT-v3" [#r.model #text(size: 20pt, fill: muted)[(#r.seeds #if r.seeds == "1" [seed] else [seeds])]] else [#r.model],
    r.hire, r.unstable, if r.delta == "" [—] else [*#r.delta*],
    text(size: 22pt, r.range), r.gpu_h)).flatten(),
)

// ---------------------------------------------------------------- findings
#let finding(number, body) = block(
  width: 100%, height: 100%, fill: white, radius: 14pt, inset: (x: 22pt, y: 16pt), {
    text(size: 58pt, weight: "bold", number)
    v(0.1em)
    text(size: 22pt, fill: muted, body)
  })

// ================================================================ page
#grid(
  rows: (auto, 9.3in, 1fr, 4.7in),
  gutter: 0.42in,

  grid(
    columns: (1fr, 6.5in),
    gutter: 0.6in,
    {
      text(size: 80pt, weight: "bold")[What training after SFT buys for stable LLM hiring decisions, and what it costs]
      v(0.1em)
      text(size: 32pt, fill: muted)[Qwen3.5-9B on an English counterfactual hiring audit, trained and evaluated on NVIDIA GH200 (CSCS Alps)]
      v(0.05em)
      text(size: 30pt)[\[Authors and affiliations, order to be agreed\]]
    },
    block(width: 100%, height: 2.6in, radius: 12pt, inset: 22pt,
      stroke: (paint: shared.line, thickness: 3pt, dash: "dashed"),
      align(center + horizon, text(size: 22pt, fill: muted)[logos])),
  ),

  grid(
    columns: (26fr, 74fr),
    gutter: 0.42in,
    section("1", [Problem formulation])[
      A model that screens CVs should give the same decision when only a protected attribute changes. After supervised fine-tuning (SFT), does further training (DPO, RL with GRPO or DAPO, or decision distillation) make the decisions more stable, and at what GPU cost?

      #v(0.2em)
      #set text(size: 27pt)
      $ U(f) = 1 / abs(cal(S)) sum_(S in cal(S)) bb(1) [exists a, b in S : f(x_a) != f(x_b)] $
      $ Delta = U(f_"method" | h) - U(f_"reference" | h) $
      #note[$S$: the attribute variants of one CV. $h$: one shared hire rate. Better means $Delta < 0$ with its 95% CI below 0.]

      #callout[#set text(size: 23pt)
        *Why one shared hire rate.* A model that rejects more looks more stable. Our GRPO pilot looked better, #N.grpo_pilot_delta pp, only because its hire rate fell #N.grpo_pilot_hire.]
    ],
    section("2", [Methodology: one pipeline for training, audit and mechanism])[
      #pipeline
    ],
  ),

  grid(
    columns: (30fr, 21fr, 49fr),
    gutter: 0.42in,
    section("3", [Dataset and an example])[
      *#N.audit_pairs* job–CV pairs (#N.audit_candidates candidates). One protected attribute is added: military status (#ex.values_per_group.military_status values), religion (#ex.values_per_group.religion) or gender (#ex.values_per_group.gender), written explicitly or implicitly, plus an attribute-free copy. *#N.audit_prompts* prompts, *#N.audit_sets* sets. Reference decision: GPT-4o.

      #example-card
    ],
    section("4", [Experiment setup])[
      #set par(spacing: 0.7em)
      *Model:* Qwen3.5-9B, English; LoRA #N.lora, merged before serving.

      *Reference:* SFT-v3, SFT on #N.vote_draws\-vote teacher decisions.

      *After SFT-v3:* DPO (4,880 on-policy pairs, #N.dpo_steps steps); GRPO and DAPO with learning-rate decay; decision distillation. Three training seeds each (one for distillation on DPO and on GRPO, and for GRPO on distillation); checkpoint declared before the audit.

      *Metric:* unstable sets at a shared hire rate, 95% CI over candidates.

      *Guardrails:* recall of GPT-4o's hires falls by at most #N.max_tpr_drop_pp pp, agreement by at most #N.max_agree_drop_pp pp; at most #N.max_undecided_pct% undecided; merge keeps at least #N.min_merge_kept_pct%.

      *Patching:* #N.patch_pairs pairs per contrast, #N.patch_contrasts contrasts.

      *Hardware:* CSCS Alps, nodes of 4 NVIDIA GH200; GPU-hours metered per step.
    ],
    section("5", [Results: every method against the base model])[
      #results-table
      #note[*vs base:* the base model moved to the same hire rate (at SFT-v3's #N.common_hire it has #N.base_at_common unstable sets). GPU-h: data and training, SFT-v3's over the base and the others on top of SFT-v3. Every row passes every guardrail.]

      #v(0.3em)
      - *SFT does almost all of it.* The best recipe after SFT-v3 adds #N.distill_mean pp (decision distillation, mean of three seeds); at SFT-v3's own hire rate no recipe is clearly better than SFT-v3.
      - *Seeds matter.* DPO's #N.dpo_seeds.at(0).own pp on seed 42 held on #N.dpo_held seeds.

      #callout(owner: marta)[*Inside the model.* In the base model the attribute reaches the decision at layers #N.pathway_layers, through head L15.H11, and is read out by MLP 31. SFT-v3 removes #N.sft_v3_effect_removed of the effect but keeps the route; DPO and GRPO widen all margins (#N.dpo_scale, #N.grpo_scale) rather than cut it.]

      #callout(owner: mine)[*The audit as a workload.* vLLM's prefix cache makes the audit of #N.audit_prompts prompts #N.b_speedup faster on one GH200.]
    ],
  ),

  block(width: 100%, height: 100%, fill: shared.bg, radius: 16pt, inset: 22pt,
    grid(
      columns: (1fr, 4.4in),
      column-gutter: 0.4in,
      grid(
        rows: (auto, 1fr),
        row-gutter: 16pt,
        {
          box(width: 46pt, height: 46pt, radius: 23pt, fill: shared.fg,
            align(center + horizon, text(size: 28pt, weight: "bold", fill: white)[6]))
          h(14pt)
          text(size: 36pt, weight: "bold")[Conclusion and findings]
          h(24pt)
          text(size: 26pt)[After a strong SFT, DPO, RL and distillation add at most about half a point of stability over three seeds, and none removes the attribute's pathway.]
        },
        grid(
          columns: (1fr,) * 5,
          column-gutter: 0.3in,
          finding(N.vs_base.at(1).delta + " pp")[SFT-v3 against the base model: almost the whole gain],
          finding(N.distill_mean + " pp")[the most later training adds on top (distillation, three seeds)],
          finding(N.dpo_held)[DPO seeds that keep the single-seed gain],
          finding(N.explore_ratio)[GPU-hours to explore GRPO and DAPO against DPO],
          finding(N.b_speedup)[faster audits from vLLM prefix caching on GH200],
        ),
      ),
      align(center + horizon, {
        image("figures/qr.svg", width: 3.3in)
        v(0.1em)
        text(size: 20pt)[*Details*, every arm and sources]
      }),
    )),
)
