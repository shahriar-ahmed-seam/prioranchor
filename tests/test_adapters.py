"""Unit tests for ecosystem adapters (Diffusers and ComfyUI)."""

import json

from prioranchor.adapters.comfyui import ComfyUIAdapter
from prioranchor.adapters.diffusers import DiffusersAdapter
from prioranchor.core.models import LoRAConfig
from prioranchor.peft.lora import LoRAContainer


def test_diffusers_adapter_roundtrip():
    cfg = LoRAConfig(rank=2, alpha=4.0)
    container = LoRAContainer(cfg)
    layer = container.inject_layer("unet.attention.to_q", in_features=4, out_features=4)

    # Set custom weights
    layer.lora_b = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

    # Convert to Diffusers dictionary
    diffusers_dict = DiffusersAdapter.to_diffusers_format(container)

    expected_down_key = "lora_unet_unet_attention_to_q.lora_down.weight"
    expected_up_key = "lora_unet_unet_attention_to_q.lora_up.weight"
    expected_alpha_key = "lora_unet_unet_attention_to_q.alpha"

    assert expected_down_key in diffusers_dict
    assert expected_up_key in diffusers_dict
    assert diffusers_dict[expected_alpha_key] == 4.0

    # Create fresh container and reload from Diffusers format
    fresh_container = LoRAContainer(cfg)
    fresh_layer = fresh_container.inject_layer("unet.attention.to_q", in_features=4, out_features=4)

    DiffusersAdapter.from_diffusers_format(diffusers_dict, fresh_container)
    assert abs(fresh_layer.lora_b[0][0] - 0.1) < 1e-6
    assert abs(fresh_layer.alpha - 4.0) < 1e-6


def test_comfyui_adapter_export():
    cfg = LoRAConfig(rank=2, alpha=4.0)
    container = LoRAContainer(cfg)
    container.inject_layer("unet.attention.to_v", in_features=4, out_features=4)

    metadata = ComfyUIAdapter.create_metadata(
        subject_name="Vintage Watch",
        trigger_phrase="a photo of a [V] watch",
    )

    package = ComfyUIAdapter.export_package(container, metadata)
    assert "__metadata__" in package
    assert "weights" in package

    meta = package["__metadata__"]
    assert meta["modelspec.title"] == "PriorAnchor LoRA - Vintage Watch"
    assert meta["modelspec.trigger_phrase"] == "a photo of a [V] watch"
    tags = json.loads(meta["modelspec.tags"])
    assert "dreambooth" in tags
