"""Unit tests for PriorAnchor Typer CLI suite."""

import json
import tempfile

from typer.testing import CliRunner

from prioranchor.cli.main import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "PriorAnchor version" in result.stdout


def test_cli_train_and_json_out():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, [
            "train",
            "--steps", "10",
            "--instance-dir", f"{tmpdir}/instances",
            "--output-dir", f"{tmpdir}/checkpoints",
            "--json-out",
        ])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 10
        assert data[0]["step"] == 0
        assert "subject_loss" in data[0]


def test_cli_priors_generate():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, [
            "priors-generate",
            "backpack",
            "--count", "10",
            "--cache-dir", f"{tmpdir}/cache",
        ])
        assert result.exit_code == 0
        assert "Successfully synthesized 10 prior samples" in result.stdout


def test_cli_eval():
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 0
    assert "PriorAnchor Metric Verification" in result.stdout
    assert "Subject Fidelity" in result.stdout

    result_json = runner.invoke(app, ["eval", "--json-out"])
    assert result_json.exit_code == 0
    data = json.loads(result_json.stdout)
    assert "fidelity" in data
    assert "drift" in data


def test_cli_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export diffusers
        diff_out = f"{tmpdir}/diffusers.json"
        res_d = runner.invoke(app, ["export", "--format", "diffusers", "--output-file", diff_out])
        assert res_d.exit_code == 0
        with open(diff_out) as f:
            d_data = json.load(f)
        assert any("lora_unet_" in k for k in d_data.keys())

        # Export comfyui
        comfy_out = f"{tmpdir}/comfyui.json"
        res_c = runner.invoke(app, ["export", "--format", "comfyui", "--output-file", comfy_out])
        assert res_c.exit_code == 0
        with open(comfy_out) as f:
            c_data = json.load(f)
        assert "__metadata__" in c_data
