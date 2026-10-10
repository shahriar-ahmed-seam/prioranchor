"""FastAPI developer console server and Anti-AI-slop telemetry workbench for PriorAnchor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from prioranchor import __version__
from prioranchor.adapters.comfyui import ComfyUIAdapter
from prioranchor.adapters.diffusers import DiffusersAdapter
from prioranchor.core.models import (
    LoRAConfig,
    OptimizationConfig,
    PriorAnchorConfig,
    PriorDatasetConfig,
    SubjectDatasetConfig,
    TokenIdentifier,
)
from prioranchor.engine.trainer import PriorAnchorTrainer
from prioranchor.metrics.drift import ConceptDriftDetector
from prioranchor.metrics.fidelity import SubjectFidelityEvaluator
from prioranchor.peft.lora import LoRAContainer

_HTML_PATH = Path(__file__).parent / "dashboard.html"


class TrainRequest(BaseModel):
    base_model: str = "runwayml/stable-diffusion-v1-5"
    token: str = "[V]"
    class_noun: str = "dog"
    steps: int = Field(default=20, ge=10, le=200)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    prior_lambda: float = Field(default=1.0, ge=0.0)
    rank: int = Field(default=16, ge=1, le=128)


class EvalRequest(BaseModel):
    token: str = "[V]"
    class_noun: str = "dog"
    drift_threshold: float = 0.25
    bleed_threshold: float = 0.15


class ExportRequest(BaseModel):
    format: str = "diffusers"
    token: str = "[V]"
    class_noun: str = "dog"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="PriorAnchor Telemetry Console",
        version=__version__,
        description="Anti-AI-slop developer telemetry console and workbench.",
    )

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard() -> str:
        """Serve developer telemetry workbench."""
        if _HTML_PATH.exists():
            content = _HTML_PATH.read_text(encoding="utf-8")
            return content.replace("v0.1.0", f"v{__version__}")
        return "<h1>PriorAnchor Telemetry Console</h1>"

    @app.get("/api/status")
    async def get_status() -> Dict[str, Any]:
        """Telemetry health ping and active system state."""
        return {
            "service": "prioranchor",
            "version": __version__,
            "status": "OPERATIONAL",
            "active_framework": "Diffusion Dual-Loss Engine",
            "loss_balance_lambda": 1.0,
            "peft_adapter": "LoRA (r=16, alpha=32)",
            "supported_schedulers": ["ddpm", "ddim", "flow_matching"],
        }

    @app.post("/api/train/run")
    async def train_endpoint(req: TrainRequest) -> Dict[str, Any]:
        """Trigger simulated dual-objective training and return step telemetry."""
        config = PriorAnchorConfig(
            base_model_path=req.base_model,
            token_identifier=TokenIdentifier(
                placeholder_token=req.token,
                class_noun=req.class_noun,
            ),
            subject_dataset=SubjectDatasetConfig(
                instance_data_dir="data/instances",
                instance_prompt=f"a photo of a {req.token} {req.class_noun}",
                class_prompt=f"a photo of a {req.class_noun}",
            ),
            prior_dataset=PriorDatasetConfig(
                num_class_images=20,
                cache_dir="prior_cache",
            ),
            lora=LoRAConfig(rank=req.rank),
            optimization=OptimizationConfig(
                learning_rate=req.learning_rate,
                prior_preservation_lambda=req.prior_lambda,
                max_train_steps=req.steps,
            ),
            output_dir="checkpoints",
        )

        trainer = PriorAnchorTrainer(config)
        history = trainer.train(max_steps=req.steps)
        events = [e.model_dump() for e in history]

        return {
            "status": "COMPLETED",
            "total_steps": len(events),
            "final_subject_loss": events[-1]["subject_loss"],
            "final_prior_loss": events[-1]["prior_loss"],
            "final_total_loss": events[-1]["total_loss"],
            "history": events,
        }

    @app.post("/api/eval/run")
    async def eval_endpoint(req: EvalRequest) -> Dict[str, Any]:
        """Evaluate concept drift, language bleed, and subject fidelity."""
        ref_subj = [1.0, 0.2, 0.0, 0.1] + [0.0] * 12
        gen_subj = [0.94, 0.23, 0.04, 0.09] + [0.0] * 12
        prompt_emb = [0.86, 0.28, 0.1, 0.1] + [0.0] * 12

        fid_eval = SubjectFidelityEvaluator(embedding_dim=16)
        fid_score = fid_eval.evaluate_sample(gen_subj, ref_subj, prompt_emb)

        drift_detector = ConceptDriftDetector(
            drift_threshold=req.drift_threshold,
            bleed_threshold=req.bleed_threshold,
        )
        base_class_outs = [[0.0, 0.9, 0.1] + [0.0] * 13]
        adapted_class_outs = [[0.02, 0.88, 0.11] + [0.0] * 13]
        drift_report = drift_detector.evaluate_drift(base_class_outs, adapted_class_outs, ref_subj)

        return {
            "fidelity": fid_score.model_dump(),
            "drift": drift_report.model_dump(),
        }

    @app.post("/api/export")
    async def export_endpoint(req: ExportRequest) -> Dict[str, Any]:
        """Export adapter state dictionary."""
        container = LoRAContainer(LoRAConfig(rank=16))
        for m in ["to_q", "to_k", "to_v", "to_out.0"]:
            container.inject_layer(f"unet.attention.{m}", in_features=16, out_features=16)

        fmt = req.format.lower().strip()
        if fmt == "diffusers":
            data = DiffusersAdapter.to_diffusers_format(container)
        elif fmt == "comfyui":
            meta = ComfyUIAdapter.create_metadata(
                subject_name=f"{req.token} {req.class_noun}",
                trigger_phrase=f"a photo of a {req.token} {req.class_noun}",
            )
            data = ComfyUIAdapter.export_package(container, meta)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format '{req.format}'")

        return {
            "format": fmt,
            "keys_exported": len(data),
            "payload_sample": list(data.keys())[:5],
        }

    return app


app = create_app()
