"""Inference pipeline adapter for diffusion generation with PriorAnchor LoRA adapters."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from prioranchor.core.math import compute_snr
from prioranchor.core.models import LoRAConfig
from prioranchor.peft.lora import LoRAContainer


class GenerationStepTelemetry(BaseModel):
    """Telemetry data captured at an individual denoising timestep."""

    step: int
    timestep: float
    snr: float
    uncond_norm: float
    cond_norm: float
    guided_delta_norm: float


class GenerationOutput(BaseModel):
    """Structured generation artifact returned by PriorAnchorPipeline."""

    prompt: str
    negative_prompt: str
    num_inference_steps: int
    guidance_scale: float
    seed: int
    fused: bool
    lora_active: bool
    final_latent_mean: float
    final_latent_variance: float
    final_latent_vector: List[float] = Field(default_factory=list)
    steps_telemetry: List[GenerationStepTelemetry] = Field(default_factory=list)


class PriorAnchorPipeline:
    """Unified inference engine integrating diffusion cross-attention and PEFT adapters."""

    def __init__(
        self,
        container: Optional[LoRAContainer] = None,
        latent_dim: int = 64,
        text_dim: int = 64,
    ):
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.fused = False
        self.lora_scale = 1.0

        if container is None:
            config = LoRAConfig(rank=16, alpha=32.0)
            self.container = LoRAContainer(config)
            self._init_default_cross_attention_layers()
        else:
            self.container = container

    def _init_default_cross_attention_layers(self) -> None:
        """Initialize standard UNet cross-attention adapter targets."""
        for layer_name in [
            "unet.down_blocks.1.attentions.0.to_q",
            "unet.down_blocks.1.attentions.0.to_k",
            "unet.down_blocks.1.attentions.0.to_v",
            "unet.mid_block.attentions.0.to_q",
            "unet.mid_block.attentions.0.to_v",
            "unet.up_blocks.1.attentions.0.to_q",
            "unet.up_blocks.1.attentions.0.to_v",
        ]:
            if layer_name not in self.container.layers:
                self.container.inject_layer(
                    name=layer_name,
                    in_features=self.latent_dim,
                    out_features=self.latent_dim,
                )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> PriorAnchorPipeline:
        """Instantiate pipeline from a saved JSON checkpoint or .safetensors file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if path.suffix == ".safetensors":
            from prioranchor.adapters.diffusers import DiffusersAdapter

            # Determine rank and dimensions from the safetensors header
            from prioranchor.adapters.safetensors_io import SafeTensorsIO

            tensors, metadata = SafeTensorsIO.load_file(path)
            # Find an alpha or a lora_down matrix to infer rank
            rank = 16
            alpha = 32.0
            in_dim = 64
            for k, val in tensors.items():
                if "lora_down.weight" in k and isinstance(val, list) and len(val) > 0:
                    rank = len(val)
                    if isinstance(val[0], list):
                        in_dim = len(val[0])
                if "alpha" in k:
                    alpha = float(val) if isinstance(val, (int, float)) else float(val[0])

            config = LoRAConfig(rank=rank, alpha=alpha)
            container = LoRAContainer(config)

            # Discover layers from metadata or tensor keys
            if "prioranchor_layers" in metadata:
                layer_names = json.loads(metadata["prioranchor_layers"])
            else:
                discovered = set()
                for k in tensors:
                    if k.startswith("lora_unet_") and k.endswith(".lora_down.weight"):
                        clean = k.replace("lora_unet_", "").replace(".lora_down.weight", "")
                        orig = clean.replace("_", ".")
                        discovered.add(orig)
                layer_names = list(discovered)

            for l_name in layer_names:
                container.inject_layer(l_name, in_features=in_dim, out_features=in_dim)

            DiffusersAdapter.from_diffusers_format(tensors, container)
            return cls(container=container, latent_dim=in_dim, text_dim=in_dim)

        # Standard JSON checkpoint
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            config = LoRAConfig(rank=16, alpha=32.0)
            return cls(container=LoRAContainer(config))

        first_val = next(iter(data.values()))
        rank = int(first_val.get("rank", 16))
        alpha = float(first_val.get("alpha", 32.0))
        in_dim = int(first_val.get("in_features", 64))
        out_dim = int(first_val.get("out_features", 64))

        config = LoRAConfig(rank=rank, alpha=alpha)
        container = LoRAContainer(config)
        for name, layer_dict in data.items():
            layer = container.inject_layer(name, in_features=in_dim, out_features=out_dim)
            layer.lora_a = layer_dict["weight_a"]
            layer.lora_b = layer_dict["weight_b"]
            layer.alpha = layer_dict["alpha"]
            layer.scaling = layer.alpha / float(layer.rank)
            layer.merged = layer_dict.get("merged", False)

        return cls(container=container, latent_dim=in_dim, text_dim=in_dim)

    def fuse_lora(self) -> None:
        """Fold all active LoRA weights into base weights for zero-overhead inference."""
        self.container.merge_all()
        self.fused = True

    def unfuse_lora(self) -> None:
        """Unfold LoRA weights back into adapter matrices."""
        self.container.unmerge_all()
        self.fused = False

    def _encode_text(self, text: str, seed: int) -> List[float]:
        """Deterministic pseudo-text embedder mapping tokens to normalized text embedding."""
        rng = random.Random(f"{text}_{seed}")
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.text_dim)]
        norm = math.sqrt(sum(v * v for v in raw)) + 1e-8
        return [v / norm for v in raw]

    def _predict_noise_vector(
        self,
        latent: List[float],
        text_embed: List[float],
        timestep: float,
    ) -> List[float]:
        """Compute UNet noise prediction modulated by cross-attention LoRA layers."""
        snr = compute_snr(timestep)
        snr_factor = 1.0 / (1.0 + math.exp(-snr))

        out_noise = [0.0] * self.latent_dim

        # Forward pass across active attention layers
        for layer in self.container.layers.values():
            adapted = layer.forward(latent)
            for i in range(self.latent_dim):
                cross_term = adapted[i] * text_embed[i % self.text_dim]
                out_noise[i] += cross_term * snr_factor

        num_layers = max(1, len(self.container.layers))
        return [v / num_layers for v in out_noise]

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> GenerationOutput:
        """Execute classifier-free guidance iterative denoising loop.

        Args:
            prompt: Text prompt describing the target generation.
            negative_prompt: Unconditional conditioning prompt.
            num_inference_steps: Number of discrete diffusion steps.
            guidance_scale: Classifier-free guidance multiplier (s).
            seed: RNG seed for initial latent noise.

        Returns:
            GenerationOutput with telemetry and final latent stats.
        """
        rng = random.Random(seed)
        latent = [rng.gauss(0.0, 1.0) for _ in range(self.latent_dim)]

        cond_embed = self._encode_text(prompt, seed)
        uncond_embed = self._encode_text(negative_prompt, seed + 1000)

        telemetry: List[GenerationStepTelemetry] = []

        step_size = 1000.0 / float(num_inference_steps)

        for step in range(num_inference_steps):
            timestep = max(0.0, 1000.0 - (step * step_size))
            snr = compute_snr(timestep)

            # Predict unconditionally and conditionally
            eps_uncond = self._predict_noise_vector(latent, uncond_embed, timestep)
            eps_cond = self._predict_noise_vector(latent, cond_embed, timestep)

            # Classifier-Free Guidance: eps = eps_uncond + s * (eps_cond - eps_uncond)
            eps_guided = [
                u + guidance_scale * (c - u) for u, c in zip(eps_uncond, eps_cond)
            ]

            # Denoise step: latent = latent - alpha * eps_guided
            alpha_step = 1.0 / float(num_inference_steps)
            latent = [z - alpha_step * eg for z, eg in zip(latent, eps_guided)]

            uncond_norm = math.sqrt(sum(v * v for v in eps_uncond))
            cond_norm = math.sqrt(sum(v * v for v in eps_cond))
            delta_norm = math.sqrt(sum((c - u) ** 2 for c, u in zip(eps_cond, eps_uncond)))

            telemetry.append(
                GenerationStepTelemetry(
                    step=step + 1,
                    timestep=round(timestep, 1),
                    snr=round(snr, 4),
                    uncond_norm=round(uncond_norm, 4),
                    cond_norm=round(cond_norm, 4),
                    guided_delta_norm=round(delta_norm, 4),
                )
            )

        mean_val = sum(latent) / float(len(latent))
        var_val = sum((v - mean_val) ** 2 for v in latent) / float(len(latent))

        return GenerationOutput(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            fused=self.fused,
            lora_active=True,
            final_latent_mean=round(mean_val, 6),
            final_latent_variance=round(var_val, 6),
            final_latent_vector=[round(v, 6) for v in latent],
            steps_telemetry=telemetry,
        )
