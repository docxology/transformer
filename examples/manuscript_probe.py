"""Reproduce the synthetic CPU evidence in the Transformer manuscript."""

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.config import get_preset_config
from src.transformer import Transformer
from src.visualization import compute_path_counts, trace_information_paths, visualize_attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    output_dir = parser.parse_args().output_dir
    figures = output_dir / "figures"
    data = output_dir / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    seed = 7
    torch.manual_seed(seed)
    config = get_preset_config("debug")
    model = Transformer(config).eval()
    tokens = torch.tensor([[2, 7, 11, 3, 7, 5]])
    modified = tokens.clone()
    modified[0, 3:] = torch.tensor([13, 17, 19])

    with torch.inference_mode():
        logits, state = model(tokens, return_state=True)
        modified_logits, _ = model(modified)
        attention = [state.get_attention_weights(i) for i in range(len(state.layer_states))]

    if not all(torch.isfinite(tensor).all() for tensor in [logits, modified_logits, *attention]):
        raise RuntimeError("The probe produced non-finite logits or attention weights")

    future_mass = max(weights.triu(diagonal=1).abs().max().item() for weights in attention)
    row_error = max((weights.sum(dim=-1) - 1).abs().max().item() for weights in attention)
    prefix_error = (logits[:, :3] - modified_logits[:, :3]).abs().max().item()
    suffix_delta = (logits[:, 3:] - modified_logits[:, 3:]).abs().max().item()
    paths = compute_path_counts(2, 2)
    enumerated_paths = len(trace_information_paths(0, 0, 2, 2))

    if future_mass != 0 or row_error > 1e-6 or prefix_error > 1e-6:
        raise RuntimeError("Causal attention or prefix invariance check failed")
    if not math.isfinite(suffix_delta) or suffix_delta <= 1e-6:
        raise RuntimeError("Suffix replacement did not change the suffix logits")
    if paths != enumerated_paths:
        raise RuntimeError("Toy grid count disagrees with enumeration")

    figure = visualize_attention(
        attention[0],
        layer_idx=0,
        tokens=[str(token) for token in tokens[0].tolist()],
        figsize=(8, 3.5),
        save_path=str(figures / "causal_attention.png"),
    )
    plt.close(figure)

    result = {
        "seed": seed,
        "preset": "debug",
        "config": asdict(config),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "token_ids": tokens[0].tolist(),
        "replacement_token_ids": modified[0].tolist(),
        "logits_shape": list(logits.shape),
        "attention_shape_per_block": list(attention[0].shape),
        "residual_shapes": [list(value.shape) for value in state.get_residual_stream()],
        "max_future_weight": future_mass,
        "max_attention_row_error": row_error,
        "max_prefix_logit_delta": prefix_error,
        "max_suffix_logit_delta": suffix_delta,
        "grid_count": paths,
        "grid_enumerated_count": enumerated_paths,
    }
    (data / "causal_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
