"""Utilities for merging, unmerging, and inspecting parameter efficiency ratios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from prioranchor.peft.lora import LoRAContainer, LoRAWeights


def calculate_compression_ratio(
    total_base_params: int,
    trainable_lora_params: int,
) -> float:
    """Compute parameter reduction efficiency: trainable / total."""
    if total_base_params <= 0:
        return 0.0
    return float(trainable_lora_params) / float(total_base_params)


def save_lora_checkpoint(container: LoRAContainer, destination_path: str) -> str:
    """Serialize all LoRA weights in the container to a structured JSON format."""
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "format": "prioranchor_lora_v1",
        "rank": container.config.rank,
        "alpha": container.config.alpha,
        "layers": container.state_dict(),
    }

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(dest)


def load_lora_checkpoint(container: LoRAContainer, source_path: str) -> None:
    """Load and inject LoRA weights from serialized file into existing container."""
    with open(source_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    layers_dict = data.get("layers", {})
    for name, layer_data in layers_dict.items():
        if name in container.layers:
            weights = LoRAWeights.model_validate(layer_data)
            container.layers[name].load_weights(weights)
