"""Dual-stream dataset abstraction interleaving subject instances and class priors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from pydantic import BaseModel

from prioranchor.core.models import SubjectDatasetConfig
from prioranchor.prior.generator import PriorSample


class PairedBatch(BaseModel):
    """Paired batch containing subject instance and class prior samples."""
    step: int
    subject_image_paths: List[str]
    subject_prompt: str
    prior_image_paths: List[str]
    prior_prompts: List[str]
    has_prior: bool = True


class DualStreamDataset:
    """Interleaves subject instances and class prior samples for dual-stream optimization."""

    def __init__(
        self,
        subject_config: SubjectDatasetConfig,
        prior_samples: Optional[List[PriorSample]] = None,
    ):
        self.subject_config = subject_config
        self.prior_samples = prior_samples or []

        # Discover instance images
        inst_dir = Path(subject_config.instance_data_dir)
        self.subject_images: List[str] = []
        if inst_dir.exists() and inst_dir.is_dir():
            for f in inst_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    self.subject_images.append(str(f))

        # Fallback dummy paths if running in mock directory
        if not self.subject_images:
            self.subject_images = [
                str(inst_dir / f"instance_{i:02d}.png") for i in range(4)
            ]

    def __len__(self) -> int:
        return max(len(self.subject_images), len(self.prior_samples) or 1)

    def iterate_batches(
        self,
        max_steps: int,
        batch_size: int = 1,
    ) -> Iterator[PairedBatch]:
        """Yield paired (subject, prior) batches for max_steps iterations."""
        num_subj = len(self.subject_images)
        num_prior = len(self.prior_samples)

        for step in range(max_steps):
            # Select subject images with wrap-around
            subj_paths = [
                self.subject_images[(step * batch_size + i) % num_subj]
                for i in range(batch_size)
            ]

            prior_paths: List[str] = []
            prior_prompts: List[str] = []

            if num_prior > 0:
                for i in range(batch_size):
                    prior_sample = self.prior_samples[(step * batch_size + i) % num_prior]
                    prior_paths.append(prior_sample.image_path)
                    prior_prompts.append(prior_sample.prompt)
            else:
                prior_paths = [self.subject_config.class_prompt] * batch_size
                prior_prompts = [self.subject_config.class_prompt] * batch_size

            yield PairedBatch(
                step=step,
                subject_image_paths=subj_paths,
                subject_prompt=self.subject_config.instance_prompt,
                prior_image_paths=prior_paths,
                prior_prompts=prior_prompts,
                has_prior=(num_prior > 0),
            )
