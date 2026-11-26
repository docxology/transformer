# Technical manuscript

The source is the ordered Markdown, renderer configuration, LaTeX
preamble, and bibliography in this directory. It is a technical account
of the local model and a synthetic CPU experiment. See the
[analysis guide](../guides/analysis.md) for the meaning of each check.

## Generate the evidence and render

Install Git, uv, Pandoc `3.14`, and XeLaTeX with the PDF dependencies listed in
the [Template setup guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/guides/startup-and-setup.md).
Run from the standalone project root with the desired manuscript commit
checked out. The detached checkout in Template's ignored working area
keeps the source commit and its rendered artifacts separate.

```bash
transformer_source="$(pwd -P)"
transformer_commit="$(git rev-parse HEAD)"
render_workspace="$(mktemp -d)"
template_checkout="$render_workspace/template"
git clone https://github.com/docxology/template.git "$template_checkout"
git -C "$template_checkout" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p "$template_checkout/projects/working"
git clone "$transformer_source" "$template_checkout/projects/working/transformer"
cd "$template_checkout/projects/working/transformer"
git checkout --detach "$transformer_commit"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu \
  -r requirements.txt 'torch==2.15.0+cpu' 'numpy==2.5.4' 'matplotlib==3.11.2' 'seaborn==0.13.2'
MPLBACKEND=Agg OMP_NUM_THREADS=1 .venv/bin/python -m examples.manuscript_probe --output-dir output
cd "$template_checkout"
uv sync --locked
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/transformer
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/transformer
```

In the working project, inspect `output/pdf/transformer_combined.pdf`,
`output/web/index.html`, and the validation and source provenance in
`output/reports/`. The native `outputs/` directory contains historical
example material and remains separate from this build.
