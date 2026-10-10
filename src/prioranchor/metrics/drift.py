"""Concept drift detector and language bleed index telemetry."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from prioranchor.metrics.fidelity import cosine_similarity


class DriftReport(BaseModel):
    """Telemetry report diagnosing concept drift and language bleed."""
    class_drift_divergence: float = Field(
        ...,
        description="Cosine distance (1 - cos_sim) between base and adapted models.",
    )
    language_bleed_index: float = Field(
        ...,
        description="Excess subject similarity in generic class outputs. Lower is better.",
    )
    prior_retention_rate: float = Field(
        ...,
        description="Estimated percentage of base class distribution retained.",
    )
    status: str = Field(
        ...,
        description="Health assessment: OPTIMAL, MARGINAL_DRIFT, or SEVERE_DEGRADATION.",
    )


class ConceptDriftDetector:
    """Monitors catastrophic forgetting and language drift in fine-tuned diffusion models."""

    def __init__(self, drift_threshold: float = 0.25, bleed_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.bleed_threshold = bleed_threshold

    def evaluate_drift(
        self,
        base_class_outputs: List[List[float]],
        adapted_class_outputs: List[List[float]],
        subject_reference_features: List[float],
    ) -> DriftReport:
        """Analyze concept drift on generic class prompts before and after adaptation."""
        if not base_class_outputs or len(base_class_outputs) != len(adapted_class_outputs):
            raise ValueError("Base and adapted output lists must be non-empty and of equal size.")

        # Compute cosine similarities between base and adapted model on identical class prompts
        class_sims = [
            cosine_similarity(b, a)
            for b, a in zip(base_class_outputs, adapted_class_outputs)
        ]
        avg_class_sim = sum(class_sims) / len(class_sims)
        class_drift = max(0.0, 1.0 - avg_class_sim)

        # Compute language bleed: similarity of generic outputs to specific subject
        base_to_subj = [
            cosine_similarity(b, subject_reference_features)
            for b in base_class_outputs
        ]
        adapted_to_subj = [
            cosine_similarity(a, subject_reference_features)
            for a in adapted_class_outputs
        ]
        avg_base_to_subj = sum(base_to_subj) / len(base_to_subj)
        avg_adapted_to_subj = sum(adapted_to_subj) / len(adapted_to_subj)

        # Excess similarity indicates semantic bleeding of [V] into generic class
        bleed_index = max(0.0, avg_adapted_to_subj - avg_base_to_subj)

        # Prior retention rate (1.0 - drift, clamped to 0.0 - 1.0)
        retention_rate = max(0.0, min(1.0, 1.0 - class_drift))

        if class_drift <= self.drift_threshold and bleed_index <= self.bleed_threshold:
            status = "OPTIMAL"
        elif class_drift <= self.drift_threshold * 1.5:
            status = "MARGINAL_DRIFT"
        else:
            status = "SEVERE_DEGRADATION"

        return DriftReport(
            class_drift_divergence=round(class_drift, 4),
            language_bleed_index=round(bleed_index, 4),
            prior_retention_rate=round(retention_rate * 100.0, 2),
            status=status,
        )
