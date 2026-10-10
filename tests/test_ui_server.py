"""Unit tests for FastAPI telemetry console and UI server endpoints."""

from fastapi.testclient import TestClient

from prioranchor.ui.server import create_app

client = TestClient(create_app())


def test_ui_dashboard_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PriorAnchor" in res.text
    assert "lossCanvas" in res.text


def test_ui_api_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert data["service"] == "prioranchor"


def test_ui_api_train_run():
    payload = {
        "base_model": "dummy-model",
        "token": "[V]",
        "class_noun": "clock",
        "steps": 10,
        "learning_rate": 1e-4,
        "prior_lambda": 1.0,
        "rank": 4,
    }
    res = client.post("/api/train/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["total_steps"] == 10
    assert "final_subject_loss" in data
    assert len(data["history"]) == 10


def test_ui_api_eval_run():
    payload = {
        "token": "[V]",
        "class_noun": "dog",
        "drift_threshold": 0.25,
        "bleed_threshold": 0.15,
    }
    res = client.post("/api/eval/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "fidelity" in data
    assert "drift" in data
    assert data["drift"]["status"] == "OPTIMAL"


def test_ui_api_export():
    payload = {
        "format": "diffusers",
        "token": "[V]",
        "class_noun": "dog",
    }
    res = client.post("/api/export", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["format"] == "diffusers"
    assert data["keys_exported"] > 0
