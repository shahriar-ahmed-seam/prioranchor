# Changelog

All notable changes to PriorAnchor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

### Added
- Core domain models, configuration schemas, and mathematical noise scheduling formulations (`prioranchor.core`).
- Synthetic prior generator, prompt template synthesizer, and SHA-256 content-addressable cache (`prioranchor.prior`).
- Modular LoRA adapter layers, LoRA container, and weight merger (`prioranchor.peft`).
- Dual-objective prior-preservation training engine and loss orchestrator (`prioranchor.engine`).
- Quantitative concept drift detector, DINO/CLIP subject fidelity evaluator, and language bleed index (`prioranchor.metrics`).
- Ecosystem adapters for Hugging Face Diffusers pipelines, SafeTensors binary serialization, and ComfyUI ModelSpec metadata (`prioranchor.adapters`).
- Unified PriorAnchorPipeline for personalizing diffusion inference and Classifier-Free Guidance denoising (`prioranchor.adapters.pipeline`).
- SafeTensors binary serialization and deserialization engine with zero dependencies (`prioranchor.adapters.safetensors_io`).
- Full operator CLI suite built with Typer and Rich including `generate` command (`prioranchor.cli`).
- Anti-AI-slop developer telemetry workbench and FastAPI dashboard (`prioranchor.ui`).
- Validated PyPI source distribution (sdist) and binary wheel packaging.
- Comprehensive unit test suite with 100% test pass rate.

