"""Subject fidelity and prompt alignment metrics (DINO and CLIP formulation)."""

from __future__ import annotations

import math
from typing import List

from pydantic import BaseModel, Field


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two non-zero vectors."""
    if not vec_a or len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be non-empty and of matching dimension.")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    sim = dot / (norm_a * norm_b)
    return float(max(min(sim, 1.0), -1.0))


class FidelityScore(BaseModel):
    """Evaluation score quantifying personalization fidelity and prompt adherence."""
    subject_fidelity: float = Field(
        ...,
        description="Cosine similarity to reference subject embeddings (DINO / CLIP-I)."
    )
    prompt_alignment: float = Field(
        ...,
        description="Cosine similarity of generated output to target prompt embedding (CLIP-T)."
    )
    harmonic_score: float = Field(
        ...,
        description="Harmonic mean of subject fidelity and prompt alignment."
    )


class SubjectFidelityEvaluator:
    """Evaluates subject preservation and text alignment across generated instances."""

    def __init__(self, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim

    def evaluate_sample(
        self,
        generated_features: List[float],
        reference_features: List[float],
        prompt_features: List[float],
    ) -> FidelityScore:
        """Evaluate a single generated output against reference images and target text."""
        subj_fid = cosine_similarity(generated_features, reference_features)
        prompt_ali = cosine_similarity(generated_features, prompt_features)

        # Normalize cosine similarities to [0, 1] range for harmonic calculation
        norm_subj = max(subj_fid, 0.0)
        norm_prompt = max(prompt_ali, 0.0)

        if (norm_subj + norm_prompt) > 1e-6:
            harmonic = 2.0 * (norm_subj * norm_prompt) / (norm_subj + norm_prompt)
        else:
            harmonic = 0.0

        return FidelityScore(
            subject_fidelity=round(subj_fid, 4),
            prompt_alignment=round(prompt_ali, 4),
            harmonic_score=round(harmonic, 4),
        )

    def evaluate_batch(
        self,
        batch_generated: List[List[float]],
        reference_features: List[float],
        batch_prompts: List[List[float]],
    ) -> FidelityScore:
        """Evaluate an aggregate benchmark set."""
        if not batch_generated:
            raise ValueError("Evaluation batch cannot be empty.")

        scores = [
            self.evaluate_sample(gen, reference_features, prm)
            for gen, prm in zip(batch_generated, batch_prompts)
        ]

        avg_subj = sum(s.subject_fidelity for s in scores) / len(scores)
        avg_prompt = sum(s.prompt_alignment for s in scores) / len(scores)

        norm_s = max(avg_subj, 0.0)
        norm_p = max(avg_prompt, 0.0)
        harmonic = 2.0 * (norm_s * norm_p) / (norm_s + norm_p) if (norm_s + norm_p) > 1e-6 else 0.0

        return FidelityScore(
            subject_fidelity=round(avg_subj, 4),
            prompt_alignment=round(avg_prompt, 4),
            harmonic_score=round(harmonic, 4),
        )
