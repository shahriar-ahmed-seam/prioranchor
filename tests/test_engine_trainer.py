"""Unit tests for training engine and loss orchestrator."""

import tempfile

from prioranchor.core.models import (
    LoRAConfig,
    LossWeightingSchedule,
    OptimizationConfig,
    PriorAnchorConfig,
    PriorDatasetConfig,
    SubjectDatasetConfig,
    TelemetryEvent,
    TokenIdentifier,
)
from prioranchor.engine.loss import LossOrchestrator
from prioranchor.engine.trainer import PriorAnchorTrainer


def test_loss_orchestrator():
    opt_cfg = OptimizationConfig(
        prior_preservation_lambda=1.0,
        lambda_warmup_steps=10,
        loss_schedule=LossWeightingSchedule.MIN_SNR_GAMMA,
    )
    orch = LossOrchestrator(opt_cfg)

    # Step 0: lambda should be 0 due to warmup
    l_step0 = orch.get_effective_lambda(0)
    assert abs(l_step0) < 1e-4

    # Step 10: lambda should be 1.0
    l_step10 = orch.get_effective_lambda(10)
    assert abs(l_step10 - 1.0) < 1e-4

    total, subj, prior, eff_l = orch.compute_step_loss(
        step=10,
        subj_pred=[0.5, 0.5],
        subj_target=[0.0, 0.0],
        prior_pred=[0.2, 0.2],
        prior_target=[0.0, 0.0],
        alpha_prod_t=0.5,
    )
    assert total > 0
    assert subj > 0
    assert prior > 0
    assert abs(eff_l - 1.0) < 1e-4


def test_prior_anchor_trainer_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = PriorAnchorConfig(
            base_model_path="dummy/model",
            token_identifier=TokenIdentifier(placeholder_token="[V]", class_noun="clock"),
            subject_dataset=SubjectDatasetConfig(
                instance_data_dir=f"{tmpdir}/subject",
                instance_prompt="a photo of a [V] clock",
                class_prompt="a photo of a clock",
            ),
            prior_dataset=PriorDatasetConfig(
                num_class_images=10,
                cache_dir=f"{tmpdir}/cache",
            ),
            lora=LoRAConfig(rank=2, target_modules=["to_q"]),
            optimization=OptimizationConfig(
                learning_rate=1e-3,
                prior_preservation_lambda=1.0,
                max_train_steps=15,
                lambda_warmup_steps=5,
            ),
            output_dir=f"{tmpdir}/output",
        )

        callbacks: list[TelemetryEvent] = []

        def on_step(e: TelemetryEvent) -> None:
            callbacks.append(e)

        trainer = PriorAnchorTrainer(cfg, on_step_callback=on_step)
        history = trainer.train(max_steps=15)

        assert len(history) == 15
        assert len(callbacks) == 15
        assert history[0].step == 0
        assert history[-1].step == 14
        assert history[0].total_loss > 0
        assert history[-1].lambda_weight == 1.0
