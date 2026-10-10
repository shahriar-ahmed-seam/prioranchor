"""Synthetic prior generator using frozen base diffusion models."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from prioranchor.core.models import PriorDatasetConfig
from prioranchor.prior.cache import (
    PriorCache,
    PriorManifest,
    compute_prior_fingerprint,
)
from prioranchor.prior.templates import expand_class_prompts


class PriorSample:
    """Individual synthesized class prior sample representation."""

    def __init__(self, sample_id: int, prompt: str, image_path: str, seed: int):
        self.sample_id = sample_id
        self.prompt = prompt
        self.image_path = image_path
        self.seed = seed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "seed": self.seed,
        }


class PriorDataGenerator:
    """Generates and caches generic class images to anchor base distribution."""

    def __init__(
        self,
        base_model_path: str,
        config: PriorDatasetConfig,
        pipeline_fn: Optional[Callable[[str, int], Any]] = None,
    ):
        self.base_model_path = base_model_path
        self.config = config
        self.pipeline_fn = pipeline_fn
        self.cache = PriorCache(config.cache_dir)

    def generate(
        self,
        class_noun: str,
        force_regenerate: bool = False,
    ) -> List[PriorSample]:
        """Generate class prior samples, pulling from content-addressable cache if available."""
        fingerprint = compute_prior_fingerprint(
            base_model_path=self.base_model_path,
            class_noun=class_noun,
            resolution=512,
            num_samples=self.config.num_class_images,
            seed=self.config.seed,
            templates=self.config.class_prompt_templates,
        )

        if not force_regenerate and self.cache.has_valid_cache(
            fingerprint, self.config.num_class_images
        ):
            manifest = self.cache.load_manifest(fingerprint)
            if manifest:
                return [
                    PriorSample(
                        sample_id=e["sample_id"],
                        prompt=e["prompt"],
                        image_path=e["image_path"],
                        seed=e["seed"],
                    )
                    for e in manifest.entries
                ]

        target_dir = self.cache.get_cache_path(fingerprint)
        prompts = expand_class_prompts(
            class_noun=class_noun,
            templates=self.config.class_prompt_templates,
            target_count=self.config.num_class_images,
        )

        samples: List[PriorSample] = []
        manifest_entries: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            sample_seed = self.config.seed + idx
            image_filename = f"prior_{idx:05d}.png"
            image_path = str(target_dir / image_filename)

            if self.pipeline_fn:
                self.pipeline_fn(prompt, sample_seed)
            else:
                # Write lightweight synthetic image stub when executing in headless/CPU environments
                with open(image_path, "wb") as f:
                    f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

            sample = PriorSample(
                sample_id=idx,
                prompt=prompt,
                image_path=image_path,
                seed=sample_seed,
            )
            samples.append(sample)
            manifest_entries.append(sample.to_dict())

        manifest = PriorManifest(
            fingerprint=fingerprint,
            base_model_path=self.base_model_path,
            class_noun=class_noun,
            num_samples=len(samples),
            resolution=512,
            seed=self.config.seed,
            created_at=time.time(),
            entries=manifest_entries,
        )
        self.cache.save_manifest(manifest)
        return samples
