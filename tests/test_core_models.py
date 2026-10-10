"""Unit tests for PriorAnchor core data models and configuration contracts."""

import pytest

from prioranchor.core.models import (
    AdapterType,
    LoRAConfig,
    PriorAnchorConfig,
    SubjectDatasetConfig,
    TelemetryEvent,
    TokenIdentifier,
)


def test_token_identifier_valid():
    token = TokenIdentifier(
        placeholder_token="[V]",
        class_noun="dog",
        initializer_token="canine"
    )
    assert token.placeholder_token == "[V]"
    assert token.class_noun == "dog"
    assert token.initializer_token == "canine"


def test_token_identifier_invalid():
    with pytest.raises(ValueError):
        TokenIdentifier(placeholder_token="   ", class_noun="dog")

    with pytest.raises(ValueError):
        TokenIdentifier(placeholder_token="[V]", class_noun="")


def test_lora_config_scaling():
    lora = LoRAConfig(rank=16, alpha=32.0)
    assert lora.scaling == 2.0
    assert lora.adapter_type == AdapterType.LORA


def test_prior_anchor_config_validation():
    token = TokenIdentifier(placeholder_token="<sks>", class_noun="backpack")
    subj = SubjectDatasetConfig(
        instance_data_dir="/tmp/backpack",
        instance_prompt="a photo of a <sks> backpack",
        class_prompt="a photo of a backpack"
    )
    cfg = PriorAnchorConfig(
        base_model_path="runwayml/stable-diffusion-v1-5",
        token_identifier=token,
        subject_dataset=subj
    )
    assert cfg.token_identifier.placeholder_token == "<sks>"
    assert cfg.prior_dataset.num_class_images == 200
    assert cfg.optimization.prior_preservation_lambda == 1.0


def test_instance_prompt_missing_token_fails():
    token = TokenIdentifier(placeholder_token="[V]", class_noun="sunglasses")
    subj = SubjectDatasetConfig(
        instance_data_dir="/tmp/sunglasses",
        instance_prompt="a photo of sunglasses",  # missing [V]
        class_prompt="a photo of sunglasses"
    )
    with pytest.raises(ValueError, match="must contain placeholder token"):
        PriorAnchorConfig(
            base_model_path="stable-diffusion",
            token_identifier=token,
            subject_dataset=subj
        )


def test_telemetry_event_serialization():
    ev = TelemetryEvent(
        step=100,
        subject_loss=0.045,
        prior_loss=0.041,
        total_loss=0.086,
        learning_rate=1e-4,
        grad_norm=0.82,
        lambda_weight=1.0
    )
    data = ev.model_dump()
    assert data["step"] == 100
    assert data["total_loss"] == 0.086
    assert "timestamp" in data
