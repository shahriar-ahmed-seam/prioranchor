"""Unit tests for prior templates, content-addressable cache, and dual-stream dataset."""

import tempfile
from pathlib import Path

from prioranchor.core.models import PriorDatasetConfig, SubjectDatasetConfig
from prioranchor.prior.cache import (
    PriorCache,
    PriorManifest,
    compute_prior_fingerprint,
)
from prioranchor.prior.dataset import DualStreamDataset
from prioranchor.prior.generator import PriorDataGenerator, PriorSample
from prioranchor.prior.templates import expand_class_prompts


def test_expand_class_prompts():
    prompts = expand_class_prompts("dog", target_count=5)
    assert len(prompts) == 5
    for p in prompts:
        assert "dog" in p


def test_prior_cache_fingerprint_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PriorCache(tmpdir)
        fp = compute_prior_fingerprint(
            base_model_path="sd-1-5",
            class_noun="dog",
            resolution=512,
            num_samples=10,
            seed=42,
        )
        assert len(fp) == 16
        assert not cache.has_valid_cache(fp, expected_count=10)

        manifest = PriorManifest(
            fingerprint=fp,
            base_model_path="sd-1-5",
            class_noun="dog",
            num_samples=10,
            resolution=512,
            seed=42,
            created_at=123456.0,
            entries=[
                {"sample_id": i, "prompt": f"dog {i}", "image_path": f"/p/{i}", "seed": i}
                for i in range(10)
            ],
        )
        cache.save_manifest(manifest)
        assert cache.has_valid_cache(fp, expected_count=10)

        loaded = cache.load_manifest(fp)
        assert loaded is not None
        assert loaded.class_noun == "dog"
        assert len(loaded.entries) == 10


def test_prior_data_generator_caching():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = PriorDatasetConfig(num_class_images=12, cache_dir=tmpdir, seed=100)
        generator = PriorDataGenerator("base-model", cfg)

        # First run generates samples
        samples1 = generator.generate("clock")
        assert len(samples1) == 12
        for s in samples1:
            assert Path(s.image_path).exists()

        # Second run hits cache
        samples2 = generator.generate("clock")
        assert len(samples2) == 12
        assert samples1[0].image_path == samples2[0].image_path


def test_dual_stream_dataset_interleaving():
    subj_cfg = SubjectDatasetConfig(
        instance_data_dir="/tmp/dummy_subject",
        instance_prompt="a photo of a [V] clock",
        class_prompt="a photo of a clock",
    )
    priors = [
        PriorSample(sample_id=i, prompt=f"a clock {i}", image_path=f"/p_{i}.png", seed=i)
        for i in range(5)
    ]
    ds = DualStreamDataset(subj_cfg, priors)
    batches = list(ds.iterate_batches(max_steps=10, batch_size=1))
    assert len(batches) == 10
    assert batches[0].has_prior is True
    assert "a [V] clock" in batches[0].subject_prompt
    assert "a clock 0" in batches[0].prior_prompts[0]
    # Check cyclic indexing
    assert batches[5].prior_prompts[0] == "a clock 0"
