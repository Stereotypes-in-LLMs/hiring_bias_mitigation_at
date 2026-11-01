# GTC 2027 poster

Two versions. `latex/main.tex` is the approved v4 layout (43 x 24 in, the official GTC 2027
format), one self-contained file: `latex/latex_data.py` (shared venv) writes the measured values
into its GENERATED DATA block and checks every run against `reports/reference_metrics.csv`.
Upload `main.tex` alone to Overleaf, or run `pdflatex` twice. It has not been compiled on Daint
(no TeX there). The Typst version below (44 x 35 in) came first.

## Typst version

A 44 x 35 in landscape poster in Typst, in six sections: problem formulation, methodology (one
diagram of the post-training and audit pipeline and Marta Samoilenko's activation-patching
pipeline), dataset with one example, experiment setup, results against the base model, and
conclusion and findings. The poster keeps the core; the QR code leads to `DETAILS.md`, which
has every recipe, the seeds, the charts, the mechanism and system results, the limitations and
the source of every number.

```bash
# needs pandas and the raw audits: the shared venv
HBM_OUTPUT_ROOT=$XW ../venv/bin/python poster/pick_example.py    # -> example.json
HBM_OUTPUT_ROOT=$XW ../venv/bin/python poster/base_compare.py    # -> base_compare.json
# needs typst, qrcode and pillow: $XW/.typst-venv
$XW/.typst-venv/bin/python poster/build_poster.py --preview /path/to/preview.png
```

`build_poster.py` reads the result tables and the two JSON files, writes every number into
`numbers.json` with the file it comes from, draws the charts (`charts.py`, used on the details
page) and the QR code into `figures/`, writes `DETAILS.md` (`details.py`) and compiles
`poster.typ` to `poster.pdf`. The layout never contains a typed number. The font is Liberation
Sans. Blue is the post-training, audit and cost work; violet is Marta's.

## Still to do

- Push main: the QR code points to `poster/DETAILS.md` on the org repository's main branch.
- Logos (dashed box).
- Agree on the title, the authors and their order (in square brackets).
- Marta to check the patching lane of the diagram and the "Inside the model" box.
- Check the numbers in `DOC_NUMBERS` against the cards they name. Those are typed from
  hand-written docs; every other number is read from a CSV or JSON.
- Confirm the GTC 2027 poster size. 44 x 35 in at 300 dpi comes from GTC Europe's poster spec.
