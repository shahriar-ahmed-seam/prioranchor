"""Dual-objective prior-preservation training engine."""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional

from prioranchor.core.models import PriorAnchorConfig, TelemetryEvent
from prioranchor.engine.loss import LossOrchestrator
from prioranchor.peft.lora import LoRAContainer
from prioranchor.peft.merger import save_lora_checkpoint
from prioranchor.prior.dataset import DualStreamDataset
from prioranchor.prior.generator import PriorDataGenerator, PriorSample


class PriorAnchorTrainer:
    """Coordinates dual-objective training loop anchoring fine-tuned models to class priors."""

    def __init__(
        self,
        config: PriorAnchorConfig,
        on_step_callback: Optional[Callable[[TelemetryEvent], None]] = None,
    ):
        self.config = config
        self.on_step_callback = on_step_callback
        self.loss_orchestrator = LossOrchestrator(config.optimization)

        # Initialize LoRA container with standard attention projections
        self.lora_container = LoRAContainer(config.lora)
        for mod in config.lora.target_modules:
            self.lora_container.inject_layer(
                name=f"unet.attention.{mod}",
                in_features=16,
                out_features=16,
            )

        self.telemetry_history: List[TelemetryEvent] = []

    def prepare_dataset(self) -> DualStreamDataset:
        """Synthesize priors if needed and build dual-stream dataset."""
        prior_generator = PriorDataGenerator(
            base_model_path=self.config.base_model_path,
            config=self.config.prior_dataset,
        )
        prior_samples: List[PriorSample] = prior_generator.generate(
            class_noun=self.config.token_identifier.class_noun
        )
        return DualStreamDataset(
            subject_config=self.config.subject_dataset,
            prior_samples=prior_samples,
        )

    def train(self, max_steps: Optional[int] = None) -> List[TelemetryEvent]:
        """Execute the prior-preservation optimization loop."""
        total_steps = max_steps or self.config.optimization.max_train_steps
        dataset = self.prepare_dataset()

        # Synthetic feature dimensions
        feat_dim = 16
        lr = self.config.optimization.learning_rate

        for batch in dataset.iterate_batches(max_steps=total_steps, batch_size=1):
            step = batch.step

            # Synthetic inputs representing subject noise and class prior noise
            # Simulating diffusion noise prediction residuals
            # Subject loss decreases with training, prior loss stays constrained
            progress = float(step) / float(max(total_steps, 1))

            subj_input = [math.sin(step + i) for i in range(feat_dim)]
            subj_target = [0.0 for _ in range(feat_dim)]

            prior_input = [math.cos(step + i) for i in range(feat_dim)]
            prior_target = [0.0 for _ in range(feat_dim)]

            # Forward passes through LoRA layers
            subj_pred = [0.0] * feat_dim
            prior_pred = [0.0] * feat_dim
            for layer in self.lora_container.layers.values():
                s_out = layer.forward(subj_input)
                p_out = layer.forward(prior_input)
                subj_pred = [sp + so for sp, so in zip(subj_pred, s_out)]
                prior_pred = [pp + po for pp, po in zip(prior_pred, p_out)]

            # Compute dual-stream loss
            # Cumulative noise schedule alpha_prod_t between 0.1 and 0.9
            alpha_prod_t = 0.5 + 0.4 * math.sin(step)
            t_total, t_subj, t_prior, eff_lambda = self.loss_orchestrator.compute_step_loss(
                step=step,
                subj_pred=subj_pred,
                subj_target=subj_target,
                prior_pred=prior_pred,
                prior_target=prior_target,
                alpha_prod_t=alpha_prod_t,
                timestep=int(500 * (1.0 - progress)),
            )

            # Simulated gradient step adjusting LoRA B parameters towards target
            grad_scale = lr * (1.0 / (1.0 + step * 0.01))
            grad_norm = 0.0
            for layer in self.lora_container.layers.values():
                for r in range(layer.rank):
                    for o in range(layer.out_features):
                        delta_b = grad_scale * (subj_pred[o] + eff_lambda * prior_pred[o])
                        layer.lora_b[o][r] -= min(max(delta_b, -0.01), 0.01)
                        grad_norm += delta_b ** 2
            grad_norm = math.sqrt(grad_norm)

            event = TelemetryEvent(
                step=step,
                subject_loss=float(t_subj),
                prior_loss=float(t_prior),
                total_loss=float(t_total),
                learning_rate=float(lr),
                grad_norm=float(grad_norm),
                lambda_weight=float(eff_lambda),
                timestamp=time.time(),
            )
            self.telemetry_history.append(event)

            if self.on_step_callback:
                self.on_step_callback(event)

        # Save checkpoint upon completion
        ckpt_path = f"{self.config.output_dir}/prioranchor_lora_final.json"
        save_lora_checkpoint(self.lora_container, ckpt_path)

        return self.telemetry_history
