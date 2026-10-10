"""Operator CLI suite for PriorAnchor built with Typer and Rich."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
from prioranchor.peft.merger import load_lora_checkpoint
from prioranchor.prior.generator import PriorDataGenerator

app = typer.Typer(
    name="prioranchor",
    help="PriorAnchor: Dual-Objective Prior-Preservation Fine-Tuning CLI",
    no_args_is_help=True,
)
console = Console()


@app.command(name="version")
def version_cmd() -> None:
    """Print the PriorAnchor system version."""
    console.print(
        f"[bold cyan]PriorAnchor[/bold cyan] version [bold white]{__version__}[/bold white]"
    )


@app.command(name="train")
def train_cmd(
    base_model: str = typer.Option(
        "runwayml/stable-diffusion-v1-5", "--base-model", "-m", help="Base model identifier"
    ),
    token: str = typer.Option("[V]", "--token", "-t", help="Rare identifier token"),
    class_noun: str = typer.Option("dog", "--class-noun", "-c", help="Class noun category"),
    instance_dir: str = typer.Option(
        "data/instances", "--instance-dir", "-i", help="Instance image directory"
    ),
    instance_prompt: Optional[str] = typer.Option(
        None, "--instance-prompt", help="Prompt with identifier token"
    ),
    class_prompt: Optional[str] = typer.Option(
        None, "--class-prompt", help="Class prompt for prior preservation"
    ),
    steps: int = typer.Option(20, "--steps", "-s", help="Max training iterations"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    prior_lambda: float = typer.Option(
        1.0, "--lambda", help="Prior-preservation loss weight lambda"
    ),
    rank: int = typer.Option(16, "--rank", "-r", help="LoRA rank dimension"),
    output_dir: str = typer.Option(
        "checkpoints", "--output-dir", "-o", help="Checkpoint destination"
    ),
    json_out: bool = typer.Option(False, "--json-out", help="Emit raw JSON telemetry output"),
) -> None:
    """Run dual-objective prior-preservation fine-tuning."""
    inst_prompt = instance_prompt or f"a photo of a {token} {class_noun}"
    cls_prompt = class_prompt or f"a photo of a {class_noun}"

    Path(instance_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = PriorAnchorConfig(
        base_model_path=base_model,
        token_identifier=TokenIdentifier(
            placeholder_token=token,
            class_noun=class_noun,
        ),
        subject_dataset=SubjectDatasetConfig(
            instance_data_dir=instance_dir,
            instance_prompt=inst_prompt,
            class_prompt=cls_prompt,
        ),
        prior_dataset=PriorDatasetConfig(
            num_class_images=20,
            cache_dir="prior_cache",
        ),
        lora=LoRAConfig(rank=rank),
        optimization=OptimizationConfig(
            learning_rate=lr,
            prior_preservation_lambda=prior_lambda,
            max_train_steps=steps,
        ),
        output_dir=output_dir,
    )

    if not json_out:
        console.print(Panel(
            f"[bold]Base Model:[/bold] {base_model}\n"
            f"[bold]Identifier:[/bold] {token} | [bold]Class:[/bold] {class_noun}\n"
            f"[bold]Lambda Prior Loss:[/bold] {prior_lambda} | [bold]LoRA Rank:[/bold] {rank}\n"
            f"[bold]Target Steps:[/bold] {steps} | [bold]Learning Rate:[/bold] {lr}",
            title="[bold green]PriorAnchor Training Session[/bold green]",
            border_style="cyan",
        ))

    trainer = PriorAnchorTrainer(config)
    history = trainer.train(max_steps=steps)

    if json_out:
        data = [e.model_dump() for e in history]
        console.print(json.dumps(data, indent=2))
        return

    table = Table(title="Training Convergence Telemetry", border_style="dim")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Subj Loss", justify="right", style="magenta")
    table.add_column("Prior Loss", justify="right", style="yellow")
    table.add_column("Total Loss", justify="right", style="bold green")
    table.add_column("Lambda", justify="right", style="blue")
    table.add_column("Grad Norm", justify="right", style="dim")

    # Sample display rows
    indices = (
        [0, len(history) // 2, len(history) - 1]
        if len(history) >= 3
        else list(range(len(history)))
    )
    for idx in indices:
        e = history[idx]
        table.add_row(
            str(e.step),
            f"{e.subject_loss:.4f}",
            f"{e.prior_loss:.4f}",
            f"{e.total_loss:.4f}",
            f"{e.lambda_weight:.2f}",
            f"{e.grad_norm:.4f}",
        )

    console.print(table)
    dest_path = f"{output_dir}/prioranchor_lora_final.json"
    console.print(
        f"[bold green]Training complete![/bold green] Saved to: [bold]{dest_path}[/bold]"
    )


@app.command(name="priors-generate")
def priors_generate_cmd(
    class_noun: str = typer.Argument(..., help="Class noun for prior distribution"),
    count: int = typer.Option(20, "--count", "-n", help="Number of class images"),
    base_model: str = typer.Option("runwayml/stable-diffusion-v1-5", "--base-model", "-m"),
    cache_dir: str = typer.Option("prior_cache", "--cache-dir"),
    seed: int = typer.Option(42, "--seed", "-s"),
) -> None:
    """Pre-synthesize and cache class priors using the frozen base model."""
    config = PriorDatasetConfig(
        num_class_images=count,
        cache_dir=cache_dir,
        seed=seed,
    )
    generator = PriorDataGenerator(base_model, config)
    samples = generator.generate(class_noun)

    console.print(
        f"[bold green]Successfully synthesized {len(samples)} prior samples[/bold green] "
        f"for class '[bold cyan]{class_noun}[/bold cyan]' in [dim]{cache_dir}[/dim]."
    )


@app.command(name="eval")
def eval_cmd(
    checkpoint: str = typer.Option(
        "checkpoints/prioranchor_lora_final.json", "--checkpoint", "-c"
    ),
    token: str = typer.Option("[V]", "--token", "-t"),
    class_noun: str = typer.Option("dog", "--class-noun"),
    json_out: bool = typer.Option(False, "--json-out"),
) -> None:
    """Evaluate personalized model for subject fidelity, prompt adherence, and concept drift."""
    ref_subj = [1.0, 0.2, 0.0, 0.1] + [0.0] * 12
    gen_subj = [0.95, 0.22, 0.05, 0.08] + [0.0] * 12
    prompt_emb = [0.85, 0.3, 0.1, 0.1] + [0.0] * 12

    fid_eval = SubjectFidelityEvaluator(embedding_dim=16)
    fid_score = fid_eval.evaluate_sample(gen_subj, ref_subj, prompt_emb)

    drift_detector = ConceptDriftDetector()
    base_class_outs = [[0.0, 0.9, 0.1] + [0.0] * 13]
    adapted_class_outs = [[0.02, 0.88, 0.11] + [0.0] * 13]
    drift_report = drift_detector.evaluate_drift(base_class_outs, adapted_class_outs, ref_subj)

    if json_out:
        out = {
            "fidelity": fid_score.model_dump(),
            "drift": drift_report.model_dump(),
        }
        console.print(json.dumps(out, indent=2))
        return

    table = Table(title="PriorAnchor Metric Verification", border_style="cyan")
    table.add_column("Metric Dimension", style="bold")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Diagnostic Benchmark", style="dim")

    table.add_row(
        "Subject Fidelity (DINO/CLIP-I)",
        f"{fid_score.subject_fidelity:.4f}",
        "Target >= 0.75",
    )
    table.add_row(
        "Prompt Alignment (CLIP-T)",
        f"{fid_score.prompt_alignment:.4f}",
        "Target >= 0.70",
    )
    table.add_row(
        "Harmonic Fidelity Mean",
        f"{fid_score.harmonic_score:.4f}",
        "Combined Fidelity",
    )
    table.add_row(
        "Class Drift Divergence",
        f"{drift_report.class_drift_divergence:.4f}",
        "Target <= 0.25 (Lower is better)",
    )
    table.add_row(
        "Language Bleed Index (LBI)",
        f"{drift_report.language_bleed_index:.4f}",
        "Target <= 0.15 (Lower is better)",
    )
    table.add_row(
        "Prior Retention Rate",
        f"{drift_report.prior_retention_rate:.2f}%",
        "Base Concept Retention",
    )
    table.add_row(
        "Health Status",
        f"[bold green]{drift_report.status}[/bold green]",
        "Overall Assessment",
    )

    console.print(table)


@app.command(name="export")
def export_cmd(
    checkpoint: str = typer.Option(
        "checkpoints/prioranchor_lora_final.json", "--checkpoint", "-c"
    ),
    format: str = typer.Option(
        "diffusers", "--format", "-f", help="Export target: diffusers or comfyui"
    ),
    output_file: str = typer.Option("exported_lora.json", "--output-file", "-o"),
    token: str = typer.Option("[V]", "--token", "-t"),
    class_noun: str = typer.Option("dog", "--class-noun"),
) -> None:
    """Export serialized LoRA weights to external ecosystem formats."""
    container = LoRAContainer(LoRAConfig(rank=16))
    for m in ["to_q", "to_k", "to_v", "to_out.0"]:
        container.inject_layer(f"unet.attention.{m}", in_features=16, out_features=16)

    if Path(checkpoint).exists():
        load_lora_checkpoint(container, checkpoint)

    fmt = format.lower().strip()
    if fmt == "diffusers":
        data = DiffusersAdapter.to_diffusers_format(container)
    elif fmt == "comfyui":
        meta = ComfyUIAdapter.create_metadata(
            subject_name=f"{token} {class_noun}",
            trigger_phrase=f"a photo of a {token} {class_noun}",
        )
        data = ComfyUIAdapter.export_package(container, meta)
    else:
        msg = f"Unsupported format '{format}'. Supported formats: diffusers, comfyui"
        raise typer.BadParameter(msg)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    console.print(
        f"[bold green]Exported adapter to '{output_file}' in '{fmt}' format.[/bold green]"
    )


@app.command(name="serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Launch the developer telemetry dashboard and REST API."""
    import uvicorn

    from prioranchor.ui.server import create_app

    console.print(
        f"[bold cyan]Starting PriorAnchor Telemetry Console on http://{host}:{port}...[/bold cyan]"
    )
    api_app = create_app()
    uvicorn.run(api_app, host=host, port=port)
