"""Quickstart example demonstrating dual-objective subject fine-tuning."""

from __future__ import annotations

import tempfile

from prioranchor.core.models import (
    LoRAConfig,
    OptimizationConfig,
    PriorAnchorConfig,
    PriorDatasetConfig,
    SubjectDatasetConfig,
    TokenIdentifier,
)
from prioranchor.engine.trainer import PriorAnchorTrainer


def main() -> None:
    print("[PriorAnchor] Starting Quickstart Training Session...")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = PriorAnchorConfig(
            base_model_path="runwayml/stable-diffusion-v1-5",
            token_identifier=TokenIdentifier(
                placeholder_token="[V]",
                class_noun="teapot",
            ),
            subject_dataset=SubjectDatasetConfig(
                instance_data_dir=f"{tmpdir}/instances",
                instance_prompt="a photo of a [V] teapot",
                class_prompt="a photo of a teapot",
            ),
            prior_dataset=PriorDatasetConfig(
                num_class_images=15,
                cache_dir=f"{tmpdir}/prior_cache",
            ),
            lora=LoRAConfig(rank=16, alpha=32.0),
            optimization=OptimizationConfig(
                learning_rate=1e-4,
                prior_preservation_lambda=1.0,
                max_train_steps=20,
                lambda_warmup_steps=5,
            ),
            output_dir=f"{tmpdir}/checkpoints",
        )

        trainer = PriorAnchorTrainer(config)
        history = trainer.train()

        print(f"[PriorAnchor] Optimization Completed across {len(history)} steps.")
        print(f"  Final   -> Total Loss: {history[-1].total_loss:.4f}")
        print(f"  Final   -> Subj Loss: {history[-1].subject_loss:.4f}")
        print(f"  Final   -> Prior Loss: {history[-1].prior_loss:.4f}")
        print(f"  Lambda  -> Effective Prior Weight: {history[-1].lambda_weight:.2f}")


if __name__ == "__main__":
    main()
