# GTC 2027 poster, first draft

A 44 x 35 in landscape poster in Typst. `build_poster.py` reads the result tables, writes every
number into `numbers.json` with the file it comes from, has `charts.py` draw the charts into
`figures/` from the same tables, and compiles `poster.typ` to `poster.pdf`. The layout never
contains a typed number.

```bash
$XW/.typst-venv/bin/python poster/build_poster.py --preview /path/to/preview.png
```

`$XW/.typst-venv` holds `typst`, `pypdfium2` and `pillow` (the login node has no LaTeX). The
font is Liberation Sans. Charts are sized at their printed width, so their text prints at
17 to 21 pt. Colours: the dataviz reference palette's blue, orange and aqua for DPO, GRPO/DAPO
and distillation; violet for SFT-v3 in Marta's panel.

Panels: blue for training, evaluation, system and cost; violet for Marta Samoilenko's
activation-patching result on the base model and SFT-v3, redrawn from her patching CSVs
(`reports/interp/patching/{base,sft}-grid-mil_combat_civ_impl-n64-s42/real.csv`). The second D
panel applies her pipeline to the later arms.

## Still to do

- Logos and the QR code (dashed boxes). The QR code needs the public URL to point at.
- Agree on the title, the authors and their order, and the contributions line (in square
  brackets on the poster).
- Marta to check her panel and the pathway sketch.
- Check the numbers in `DOC_NUMBERS` against the cards they name. Those are typed from
  hand-written docs; every other number is read from a CSV.
- Confirm the GTC 2027 poster size. 44 x 35 in at 300 dpi comes from GTC Europe's poster spec.
