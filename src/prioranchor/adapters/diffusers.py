"""Ecosystem adapter for Hugging Face Diffusers pipeline conventions."""

from __future__ import annotations

from typing import Any, Dict

from prioranchor.peft.lora import LoRAContainer


class DiffusersAdapter:
    """Bridges PriorAnchor LoRA representations and Hugging Face Diffusers weight schemas."""

    @staticmethod
    def to_diffusers_format(container: LoRAContainer) -> Dict[str, Any]:
        """Convert container layers into standard diffusers state dictionary format.

        Format:
            lora_unet_{clean_name}.lora_down.weight -> LoRA A matrix
            lora_unet_{clean_name}.lora_up.weight   -> LoRA B matrix
            lora_unet_{clean_name}.alpha            -> scalar alpha
        """
        diffusers_dict: Dict[str, Any] = {}

        for name, layer in container.layers.items():
            clean_name = name.replace(".", "_")
            key_down = f"lora_unet_{clean_name}.lora_down.weight"
            key_up = f"lora_unet_{clean_name}.lora_up.weight"
            key_alpha = f"lora_unet_{clean_name}.alpha"

            diffusers_dict[key_down] = layer.lora_a
            diffusers_dict[key_up] = layer.lora_b
            diffusers_dict[key_alpha] = layer.alpha

        return diffusers_dict

    @staticmethod
    def from_diffusers_format(
        diffusers_dict: Dict[str, Any],
        container: LoRAContainer,
    ) -> None:
        """Inject diffusers state dictionary back into a PriorAnchor container."""
        for name, layer in container.layers.items():
            clean_name = name.replace(".", "_")
            key_down = f"lora_unet_{clean_name}.lora_down.weight"
            key_up = f"lora_unet_{clean_name}.lora_up.weight"
            key_alpha = f"lora_unet_{clean_name}.alpha"

            if key_down in diffusers_dict and key_up in diffusers_dict:
                layer.lora_a = diffusers_dict[key_down]
                layer.lora_b = diffusers_dict[key_up]
                if key_alpha in diffusers_dict:
                    layer.alpha = float(diffusers_dict[key_alpha])
                    layer.scaling = layer.alpha / float(layer.rank)

    @staticmethod
    def save_safetensors(
        container: LoRAContainer,
        filepath: str,
        metadata: Dict[str, str] | None = None,
    ) -> str:
        """Export container directly to a standard Hugging Face Diffusers .safetensors file."""
        import json

        from prioranchor.adapters.safetensors_io import SafeTensorsIO

        weights_dict = DiffusersAdapter.to_diffusers_format(container)
        meta = dict(metadata) if metadata else {"format": "pt", "engine": "prioranchor"}
        meta["prioranchor_layers"] = json.dumps(list(container.layers.keys()))
        out_path = SafeTensorsIO.save_file(weights_dict, filepath, metadata=meta)
        return str(out_path)

    @staticmethod
    def load_safetensors(
        filepath: str,
        container: LoRAContainer,
    ) -> Dict[str, str]:
        """Load weights from a .safetensors file directly into a PriorAnchor container."""
        from prioranchor.adapters.safetensors_io import SafeTensorsIO

        weights_dict, metadata = SafeTensorsIO.load_file(filepath)
        DiffusersAdapter.from_diffusers_format(weights_dict, container)
        return metadata

