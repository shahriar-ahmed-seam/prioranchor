"""Loss orchestrator managing dual-stream prior-preservation loss formulation."""

from __future__ import annotations

from typing import List, Tuple

from prioranchor.core.math import (
    calculate_prior_preservation_loss,
    compute_lambda_schedule,
    compute_loss_weight,
    compute_snr,
)
from prioranchor.core.models import OptimizationConfig


class LossOrchestrator:
    """Manages dynamic weighting and computation of dual-stream DreamBooth losses.

    L_total = L_subject + lambda(step) * L_prior
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def get_effective_lambda(self, step: int) -> float:
        """Calculate the current prior preservation loss coefficient lambda(step)."""
        return compute_lambda_schedule(
            step=step,
            base_lambda=self.config.prior_preservation_lambda,
            warmup_steps=self.config.lambda_warmup_steps,
        )

    def compute_step_loss(
        self,
        step: int,
        subj_pred: List[float],
        subj_target: List[float],
        prior_pred: List[float],
        prior_target: List[float],
        alpha_prod_t: float = 0.5,
        timestep: int = 500,
    ) -> Tuple[float, float, float, float]:
        """Compute the composite loss and constituent terms for the current step.

        Returns:
            Tuple of (total_loss, subject_loss, prior_loss, effective_lambda).
        """
        snr = compute_snr(alpha_prod_t)
        weight = compute_loss_weight(
            snr=snr,
            schedule=self.config.loss_schedule,
            min_snr_gamma=self.config.min_snr_gamma,
            timestep=timestep,
            max_timesteps=1000,
        )

        effective_lambda = self.get_effective_lambda(step)

        l_total, l_subj, l_prior = calculate_prior_preservation_loss(
            subj_pred=subj_pred,
            subj_target=subj_target,
            prior_pred=prior_pred,
            prior_target=prior_target,
            lambda_weight=effective_lambda,
            subj_weight=weight,
            prior_weight=weight,
        )

        return l_total, l_subj, l_prior, effective_lambda
