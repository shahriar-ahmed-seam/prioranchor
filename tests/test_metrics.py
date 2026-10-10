"""Unit tests for concept drift and fidelity evaluation metrics."""

from prioranchor.metrics.drift import ConceptDriftDetector
from prioranchor.metrics.fidelity import SubjectFidelityEvaluator, cosine_similarity


def test_cosine_similarity():
    # Identical vectors
    v1 = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-6

    # Orthogonal vectors
    v2 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(v1, v2)) < 1e-6

    # Inverted vectors
    v3 = [-1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v1, v3) - (-1.0)) < 1e-6


def test_subject_fidelity_evaluator():
    evaluator = SubjectFidelityEvaluator(embedding_dim=4)

    ref = [1.0, 0.0, 0.0, 0.0]
    gen = [0.9, 0.1, 0.0, 0.0]
    prompt = [0.8, 0.2, 0.0, 0.0]

    score = evaluator.evaluate_sample(gen, ref, prompt)
    assert score.subject_fidelity > 0.8
    assert score.prompt_alignment > 0.8
    assert score.harmonic_score > 0.8


def test_concept_drift_detector_optimal():
    detector = ConceptDriftDetector(drift_threshold=0.25, bleed_threshold=0.15)

    # Base and adapted outputs are close (preserved prior)
    base_outs = [[1.0, 0.5, 0.0], [0.8, 0.2, 0.1]]
    adapted_outs = [[0.95, 0.52, 0.01], [0.78, 0.22, 0.09]]
    ref_subj = [0.0, 0.0, 1.0]  # Reference subject has different features

    report = detector.evaluate_drift(base_outs, adapted_outs, ref_subj)
    assert report.class_drift_divergence < 0.1
    assert report.language_bleed_index < 0.05
    assert report.prior_retention_rate > 90.0
    assert report.status == "OPTIMAL"


def test_concept_drift_detector_severe_degradation():
    detector = ConceptDriftDetector(drift_threshold=0.25, bleed_threshold=0.15)

    # Adapted outputs have drifted drastically towards subject (severe forgetting)
    base_outs = [[1.0, 0.0, 0.0]]
    adapted_outs = [[0.0, 0.0, 1.0]]  # drifted to orthogonal space
    ref_subj = [0.0, 0.0, 1.0]

    report = detector.evaluate_drift(base_outs, adapted_outs, ref_subj)
    assert report.class_drift_divergence > 0.5
    assert report.language_bleed_index > 0.5
    assert report.status == "SEVERE_DEGRADATION"
