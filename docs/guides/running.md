# Running and configuration

Follow the [quick start](quickstart.md) to set up the CPU environment.
From the repository root, use module form for the examples so that imports
resolve against the local `src` package.

| Command | Purpose and output |
| --- | --- |
| `python scripts/verify_installation.py` | Check forward passes, captured state, path counting, and token generation |
| `python -m examples.basic_usage` | Save attention and residual illustrations to `outputs/visualizations/` |
| `python -m examples.path_analysis` | Print toy-grid counts and save `outputs/visualizations/path_density.png` |
| `python -m examples.training_example` | Train on synthetic sequences and write training diagnostics |
| `python scripts/generate_complete_documentation.py` | Write the master index, file counts, and system report under `outputs/reports/` |
| `MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m pytest -q` | Run the automated test suite headlessly |

The broader `python run.py --help` runner can install dependencies and run the
examples and generation scripts. With dependencies installed, use
`python run.py --skip-install` to skip its installer. Add `--examples-only` to
skip the later generation scripts. GIF and training examples take longer
than the smoke check. Generated reports can overwrite the tracked system
report, so run that command in a disposable checkout to preserve a snapshot.

## Configuration

Use [TransformerConfig](../../src/transformer/model.py) directly or obtain one
of the seven [presets](../../src/config/presets.py) through
`get_preset_config(name)`. Use `debug` for CPU checks and `tiny` or `small` for
larger examples. Deeper presets grow compute and memory needs. For a custom
configuration, use a hidden width divisible by the number of heads and
within the positional encoder's supported dimensions, and keep input sequence
length within `max_seq_len`.

```python
from src.config import get_preset_config, load_config, save_config

save_config(get_preset_config("debug"), "outputs/configs/debug.yaml")
config = load_config("outputs/configs/debug.yaml")
```

The config helper chooses YAML or JSON by extension. It serializes model
settings. It does not save or load model weights.

## Outputs

Native scripts use `outputs/` for animations, visualizations, configs, logs,
data, and reports. The manuscript renderer uses `output/` with a separate
figure, data, PDF, and web tree. See the [manuscript build](../manuscript/README.md)
for the exact workflow.
