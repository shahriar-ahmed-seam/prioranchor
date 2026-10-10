"""Example demonstrating LoRA injection, weight merging, and ecosystem export."""

from __future__ import annotations

import json
import tempfile

from prioranchor.adapters.comfyui import ComfyUIAdapter
from prioranchor.adapters.diffusers import DiffusersAdapter
from prioranchor.core.models import LoRAConfig
from prioranchor.peft.lora import LoRAContainer
from prioranchor.peft.merger import calculate_compression_ratio, save_lora_checkpoint


def main() -> None:
    print("[PriorAnchor] Initializing LoRA Container for Diffusion Cross-Attention...")
    cfg = LoRAConfig(rank=16, alpha=32.0)
    container = LoRAContainer(cfg)

    # Inject adaptation into typical UNet cross-attention modules
    target_projections = [
        "unet.down_blocks.0.attentions.0.to_q",
        "unet.down_blocks.0.attentions.0.to_k",
        "unet.down_blocks.0.attentions.0.to_v",
        "unet.down_blocks.0.attentions.0.to_out.0",
    ]

    for proj in target_projections:
        container.inject_layer(proj, in_features=320, out_features=320)

    trainable_params = container.get_trainable_param_count()
    base_model_params = 860_000_000  # Stable Diffusion v1.5 approx param count
    ratio = calculate_compression_ratio(base_model_params, trainable_params)

    print(f"  Trainable LoRA Parameters : {trainable_params:,}")
    print(f"  Base Model Parameters     : {base_model_params:,}")
    print(f"  Parameter Footprint Ratio : {ratio * 100.0:.4f}% of backbone")

    # Perform weight merging
    print("\n[PriorAnchor] Folding LoRA weights into base weights (merge_all)...")
    container.merge_all()
    print("  Status: All cross-attention layers merged for zero-latency inference.")

    container.unmerge_all()
    print("  Status: Unmerged back to modular low-rank adapters.")

    # Export to Diffusers and ComfyUI
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_dict = DiffusersAdapter.to_diffusers_format(container)
        diff_path = f"{tmpdir}/lora_diffusers.json"
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(diff_dict, f, indent=2)
        print(f"\n[PriorAnchor] Exported {len(diff_dict)} keys to Diffusers format: {diff_path}")

        meta = ComfyUIAdapter.create_metadata(
            subject_name="Custom Teapot",
            trigger_phrase="a photo of a [V] teapot",
        )
        comfy_pkg = ComfyUIAdapter.export_package(container, meta)
        comfy_path = f"{tmpdir}/lora_comfyui.json"
        with open(comfy_path, "w", encoding="utf-8") as f:
            json.dump(comfy_pkg, f, indent=2)
        print(f"[PriorAnchor] Exported ModelSpec artifact for ComfyUI: {comfy_path}")

        ckpt_path = f"{tmpdir}/checkpoint.json"
        save_lora_checkpoint(container, ckpt_path)
        print(f"[PriorAnchor] Serialized native PriorAnchor checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
