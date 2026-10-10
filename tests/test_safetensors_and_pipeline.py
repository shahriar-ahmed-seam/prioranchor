"""Unit and integration tests for SafeTensors I/O, PriorAnchorPipeline, and CLI generation."""

from pathlib import Path

from typer.testing import CliRunner

from prioranchor.adapters.diffusers import DiffusersAdapter
from prioranchor.adapters.pipeline import PriorAnchorPipeline
from prioranchor.adapters.safetensors_io import SafeTensorsIO
from prioranchor.cli.main import app
from prioranchor.core.models import LoRAConfig
from prioranchor.peft.lora import LoRAContainer


def test_safetensors_io_roundtrip(tmp_path: Path):
    file_path = tmp_path / "test_model.safetensors"
    tensors = {
        "unet_layer1.weight": [[1.25, 2.5], [3.75, 5.0]],
        "unet_layer1.bias": [0.1, 0.2],
        "unet_layer1.alpha": 32.0,
    }
    meta = {"format": "pt", "author": "prioranchor"}

    saved_p = SafeTensorsIO.save_file(tensors, file_path, metadata=meta)
    assert saved_p.exists()

    loaded_tensors, loaded_meta = SafeTensorsIO.load_file(saved_p)
    assert loaded_meta["author"] == "prioranchor"
    assert loaded_tensors["unet_layer1.alpha"] == 32.0
    assert abs(loaded_tensors["unet_layer1.weight"][0][0] - 1.25) < 1e-5
    assert abs(loaded_tensors["unet_layer1.weight"][1][1] - 5.0) < 1e-5
    assert len(loaded_tensors["unet_layer1.bias"]) == 2


def test_diffusers_adapter_safetensors_roundtrip(tmp_path: Path):
    config = LoRAConfig(rank=4, alpha=16.0)
    container = LoRAContainer(config)
    layer = container.inject_layer("unet.attention.to_q", in_features=8, out_features=8)
    layer.lora_b = [[0.5] * 4 for _ in range(8)]

    st_path = tmp_path / "diffusers_lora.safetensors"
    DiffusersAdapter.save_safetensors(container, str(st_path), metadata={"test": "true"})
    assert st_path.exists()

    fresh_container = LoRAContainer(config)
    fresh_container.inject_layer("unet.attention.to_q", in_features=8, out_features=8)
    meta = DiffusersAdapter.load_safetensors(str(st_path), fresh_container)

    assert meta["test"] == "true"
    reloaded_layer = fresh_container.layers["unet.attention.to_q"]
    assert abs(reloaded_layer.lora_b[0][0] - 0.5) < 1e-5
    assert abs(reloaded_layer.alpha - 16.0) < 1e-5


def test_pipeline_generate_and_fusion():
    config = LoRAConfig(rank=4, alpha=8.0)
    container = LoRAContainer(config)
    container.inject_layer("unet.attn1.to_q", in_features=16, out_features=16)

    pipe = PriorAnchorPipeline(container=container, latent_dim=16, text_dim=16)
    out_modular = pipe.generate(
        prompt="a photo of a [V] dog",
        num_inference_steps=5,
        guidance_scale=5.0,
        seed=101,
    )

    assert out_modular.num_inference_steps == 5
    assert len(out_modular.steps_telemetry) == 5
    assert len(out_modular.final_latent_vector) == 16
    assert not out_modular.fused

    # Test weight folding (fuse_lora)
    pipe.fuse_lora()
    assert pipe.fused
    out_fused = pipe.generate(
        prompt="a photo of a [V] dog",
        num_inference_steps=5,
        guidance_scale=5.0,
        seed=101,
    )
    assert out_fused.fused

    # Test unfuse
    pipe.unfuse_lora()
    assert not pipe.fused


def test_pipeline_from_safetensors_checkpoint(tmp_path: Path):
    config = LoRAConfig(rank=4, alpha=8.0)
    container = LoRAContainer(config)
    container.inject_layer("unet.mid_block.attentions.0.to_q", in_features=16, out_features=16)

    st_path = tmp_path / "model.safetensors"
    DiffusersAdapter.save_safetensors(container, str(st_path))

    pipe = PriorAnchorPipeline.from_checkpoint(st_path)
    assert "unet.mid_block.attentions.0.to_q" in pipe.container.layers
    res = pipe.generate(prompt="high-res test", num_inference_steps=3, seed=7)
    assert len(res.steps_telemetry) == 3


def test_cli_export_safetensors_and_generate(tmp_path: Path):
    runner = CliRunner()
    st_file = tmp_path / "cli_exported.safetensors"
    gen_file = tmp_path / "gen_output.json"

    # Export to safetensors
    res_export = runner.invoke(
        app,
        ["export", "--format", "safetensors", "--output-file", str(st_file)],
    )
    assert res_export.exit_code == 0
    assert st_file.exists()

    # Generate with the safetensors checkpoint
    res_gen = runner.invoke(
        app,
        [
            "generate",
            "--checkpoint",
            str(st_file),
            "--prompt",
            "a photo of [V] sks dog in garden",
            "--steps",
            "4",
            "--cfg",
            "6.0",
            "--fuse",
            "--out",
            str(gen_file),
        ],
    )
    assert res_gen.exit_code == 0
    assert gen_file.exists()
