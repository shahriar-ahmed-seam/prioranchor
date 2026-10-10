"""Unit tests for noise scheduling and loss calculations."""

import pytest

from prioranchor.core.math import (
    calculate_prior_preservation_loss,
    calculate_weighted_mse,
    compute_lambda_schedule,
    compute_loss_weight,
    compute_snr,
)
from prioranchor.core.models import LossWeightingSchedule


def test_compute_snr():
    snr_mid = compute_snr(0.5)
    assert abs(snr_mid - 1.0) < 1e-4

    snr_high = compute_snr(0.9)
    assert snr_high > 1.0

    snr_low = compute_snr(0.1)
    assert snr_low < 1.0


def test_loss_weight_schedules():
    snr = 1.0
    w_const = compute_loss_weight(snr, schedule=LossWeightingSchedule.CONSTANT)
    assert w_const == 1.0

    w_gamma = compute_loss_weight(
        0.1, schedule=LossWeightingSchedule.MIN_SNR_GAMMA, min_snr_gamma=5.0
    )
    # min(1.0, 5.0 / 0.1) = 1.0
    assert w_gamma == 1.0

    w_gamma_high = compute_loss_weight(
        10.0, schedule=LossWeightingSchedule.MIN_SNR_GAMMA, min_snr_gamma=5.0
    )
    # min(1.0, 5.0 / 10.0) = 0.5
    assert abs(w_gamma_high - 0.5) < 1e-4


def test_lambda_warmup_schedule():
    # step 0: 0.0
    l_0 = compute_lambda_schedule(0, base_lambda=1.0, warmup_steps=100)
    assert abs(l_0 - 0.0) < 1e-4

    # step 100: 1.0
    l_100 = compute_lambda_schedule(100, base_lambda=1.0, warmup_steps=100)
    assert abs(l_100 - 1.0) < 1e-4

    # beyond warmup: 1.0
    l_200 = compute_lambda_schedule(200, base_lambda=1.0, warmup_steps=100)
    assert l_200 == 1.0


def test_calculate_prior_preservation_loss():
    pred_s = [1.0, 2.0, 3.0]
    target_s = [1.1, 2.1, 2.9]
    pred_p = [0.5, 0.5, 0.5]
    target_p = [0.5, 0.6, 0.4]

    l_total, l_s, l_p = calculate_prior_preservation_loss(
        pred_s, target_s, pred_p, target_p, lambda_weight=1.0
    )
    assert l_s > 0
    assert l_p > 0
    assert abs(l_total - (l_s + l_p)) < 1e-6


def test_weighted_mse_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_weighted_mse([1.0], [1.0, 2.0])
