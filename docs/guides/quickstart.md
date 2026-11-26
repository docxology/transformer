# Quick start

Run commands from the repository root. The CPU workflow below uses Python
3.12 and uv, with the official PyTorch CPU package index. The
[requirements file](../../requirements.txt) records minimum versions.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu -r requirements.txt
source .venv/bin/activate
MPLBACKEND=Agg OMP_NUM_THREADS=1 python scripts/verify_installation.py
```

Use the `debug` preset for a first trace. It has two blocks, two heads per
block, a hidden width of 64, and a vocabulary of 100 synthetic token IDs.

```python
import torch
from src.config import get_preset_config
from src.transformer import Transformer

torch.manual_seed(7)
model = Transformer(get_preset_config("debug")).eval()
tokens = torch.tensor([[2, 7, 11, 3, 7, 5]])
with torch.inference_mode():
    logits, state = model(tokens, return_state=True)

print(logits.shape)                            # [1, 6, 100]
print(state.get_attention_weights(0).shape)    # [1, 2, 6, 6]
print(len(state.get_residual_stream()))         # 3 checkpoints
```

Pass `return_state=True` when you need intermediate tensors. `eval()` disables
dropout and `inference_mode()` avoids retaining a gradient graph. For setup
alternatives, larger presets, and output locations, see the
[running guide](running.md).
