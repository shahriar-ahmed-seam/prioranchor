"""Mathematical primitives, noise schedule formulations, and loss weighting calculations."""

from __future__ import annotations

import math
from typing import List, Tuple

from prioranchor.core.models import LossWeightingSchedule


def compute_snr(alpha_prod_t: float) -> float:
    """Compute Signal-to-Noise Ratio (SNR) at cumulative noise parameter alpha_prod_t.

    SNR(t) = alpha_t^2 / sigma_t^2 = alpha_prod_t / (1.0 - alpha_prod_t)
    """
    alpha_prod_t = max(min(alpha_prod_t, 0.9999), 0.0001)
    return alpha_prod_t / (1.0 - alpha_prod_t)


def compute_loss_weight(
    snr: float,
    schedule: LossWeightingSchedule = LossWeightingSchedule.CONSTANT,
    min_snr_gamma: float = 5.0,
    timestep: int = 500,
    max_timesteps: int = 1000,
) -> float:
    """Calculate timestep loss weighting factor w(t).

    Under Min-SNR-gamma (Hang et al., 2023), weights are clamped to gamma / SNR(t)
    to prevent gradient domination at low-frequency/high-noise stages.
    """
    if schedule == LossWeightingSchedule.CONSTANT:
        return 1.0

    if schedule == LossWeightingSchedule.MIN_SNR_GAMMA:
        clamped_snr = max(snr, 1e-6)
        return min(1.0, min_snr_gamma / clamped_snr)

    if schedule == LossWeightingSchedule.COSINE:
        fraction = float(timestep) / float(max(max_timesteps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    if schedule == LossWeightingSchedule.LINEAR_RAMP:
        fraction = float(timestep) / float(max(max_timesteps, 1))
        return max(0.1, 1.0 - 0.9 * fraction)

    return 1.0


def compute_lambda_schedule(
    step: int,
    base_lambda: float = 1.0,
    warmup_steps: int = 50,
) -> float:
    """Compute the dynamic prior-preservation coefficient lambda(step).

    Ensures smooth ramping of prior regularization so subject embeddings
    can initialize without immediate prior divergence conflict.
    """
    if base_lambda <= 0.0:
        return 0.0

    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lambda

    # Cosine ramp from 0.0 to base_lambda
    fraction = float(step) / float(warmup_steps)
    return base_lambda * 0.5 * (1.0 - math.cos(math.pi * fraction))


def calculate_weighted_mse(
    pred: List[float],
    target: List[float],
    weight: float = 1.0,
) -> float:
    """Calculate weighted mean squared error: w * (1/N) * sum((pred_i - target_i)^2)."""
    if not pred or len(pred) != len(target):
        raise ValueError("Predicted and target vectors must be non-empty and of equal dimension.")

    n = len(pred)
    sse = sum((p - t) ** 2 for p, t in zip(pred, target))
    mse = sse / float(n)
    return float(weight * mse)


def calculate_prior_preservation_loss(
    subj_pred: List[float],
    subj_target: List[float],
    prior_pred: List[float],
    prior_target: List[float],
    lambda_weight: float = 1.0,
    subj_weight: float = 1.0,
    prior_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """Calculate the dual-stream DreamBooth loss: L_total = L_subj + lambda * L_prior.

    Returns:
        Tuple of (total_loss, subject_loss, prior_loss).
    """
    l_subj = calculate_weighted_mse(subj_pred, subj_target, weight=subj_weight)
    l_prior = calculate_weighted_mse(prior_pred, prior_target, weight=prior_weight)
    l_total = l_subj + (lambda_weight * l_prior)
    return l_total, l_subj, l_prior
