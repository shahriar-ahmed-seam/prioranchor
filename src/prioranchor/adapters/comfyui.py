"""Ecosystem adapter for ComfyUI and standard ModelSpec metadata schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from prioranchor.peft.lora import LoRAContainer


class ModelSpecMetadata(BaseModel):
    """Canonical ModelSpec metadata block used by ComfyUI and civitai."""
    title: str = Field(..., description="Human-readable title for the adapted subject.")
    architecture: str = Field(default="stable-diffusion-v1-5", description="Base model family.")
    trigger_phrase: str = Field(..., description="Prompt trigger phrase containing subject token.")
    author: str = Field(default="PriorAnchor", description="Author or system name.")
    date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = Field(default_factory=lambda: ["dreambooth", "prior-preservation", "lora"])


class ComfyUIAdapter:
    """Produces ComfyUI-compatible serialized artifacts with embedded ModelSpec headers."""

    @staticmethod
    def create_metadata(
        subject_name: str,
        trigger_phrase: str,
        architecture: str = "stable-diffusion-v1-5",
    ) -> ModelSpecMetadata:
        """Create structured ModelSpec metadata."""
        return ModelSpecMetadata(
            title=f"PriorAnchor LoRA - {subject_name}",
            architecture=architecture,
            trigger_phrase=trigger_phrase,
        )

    @staticmethod
    def export_package(
        container: LoRAContainer,
        metadata: ModelSpecMetadata,
    ) -> Dict[str, Any]:
        """Bundle container weights alongside ModelSpec header for ComfyUI nodes."""
        return {
            "__metadata__": {
                "modelspec.title": metadata.title,
                "modelspec.architecture": metadata.architecture,
                "modelspec.trigger_phrase": metadata.trigger_phrase,
                "modelspec.author": metadata.author,
                "modelspec.date": metadata.date,
                "modelspec.tags": json.dumps(metadata.tags),
            },
            "weights": container.state_dict(),
        }
