import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from webapp.server import app, registry, results_reader, inference_engine
from webapp.registry import ModelRegistry
from webapp.results_reader import ResultsReader


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "device" in data


def test_whitelist_rejection(client):
    # Bad dataset name
    res1 = client.get("/api/compare?dataset=malicious_traversal/../../&budget=50")
    assert res1.status_code == 400
    assert "Invalid dataset" in res1.json()["detail"]

    # Bad budget value
    res2 = client.get("/api/compare?dataset=cub&budget=12")
    assert res2.status_code == 400
    assert "Invalid budget" in res2.json()["detail"]

    # Bad class endpoint
    res3 = client.get("/api/classes?dataset=unknown_dataset")
    assert res3.status_code == 400


def test_empty_tree_states(tmp_path):
    # Create empty directories
    empty_models = tmp_path / "saved_models"
    empty_models.mkdir()
    empty_results = tmp_path / "results"
    empty_results.mkdir()

    empty_reg = ModelRegistry(empty_models)
    empty_reader = ResultsReader(tmp_path)

    # All datasets must report has_pair = False
    ds_info = empty_reg.get_available_datasets()
    for d in ds_info:
        assert d["has_pair"] is False
        assert d["available_budgets"] == []

    # KPIs must return clean not_run state, never crash
    kpis = empty_reader.get_kpis("cifar10", 50)
    assert kpis["accuracy"]["status"] == "not_run"
    assert kpis["accuracy"]["original"] is None
    assert kpis["n_concepts"]["status"] == "not_run"

    # Curves must report not_run
    curves = empty_reader.get_curve_series("cifar10")
    assert curves["status"] == "not_run"

    # Redundancy comparison must show not_run for all datasets
    red = empty_reader.get_redundancy_comparison(50)
    assert all(r["status"] == "not_run" for r in red)


def test_missing_model_predict_clean_error(client):
    # Attempt prediction on places365 which has no phase 2 pair
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/api/predict",
        data={"dataset": "places365", "budget": "50"},
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 400
    assert "does not have a complete model pair" in res.json()["detail"]


def test_predict_bad_upload_clean_error(client):
    # Send non-image text as upload file for a valid dataset
    res = client.post(
        "/api/predict",
        data={"dataset": "cub", "budget": "50"},
        files={"file": ("test.txt", io.BytesIO(b"Hello world not an image"), "text/plain")},
    )
    assert res.status_code == 400
    assert "Corrupt or unreadable" in res.json()["detail"] or "Unsupported" in res.json()["detail"]


def test_stub_model_fixture_api(tmp_path):
    # Construct a minimal sample fixture tree
    models_dir = tmp_path / "original_lfcbm" / "saved_models"
    results_dir = tmp_path / "results" / "phase2" / "cifar10"
    results_dir.mkdir(parents=True)

    # Fake result file
    fake_run = {
        "summary": {
            "dataset": "cifar10", "method": "greedy", "seed": 0, "keep_frac": 0.5,
            "acc_val": 0.85, "n_concepts": 60, "redundancy_nn_abs_corr": 0.35, "relevance_mean": 0.70
        },
        "removed_by_filter": [
            {"concept": "concept_a", "relevance": 0.2, "nearest_kept": "concept_b", "corr_with_nearest_kept": 0.88}
        ]
    }
    (results_dir / "cifar10_greedy_k50_s0.json").write_text(json.dumps(fake_run))

    reader = ResultsReader(tmp_path)
    removed = reader.get_removed_concepts("cifar10", 50)
    assert len(removed) == 1
    assert removed[0]["concept"] == "concept_a"
    assert removed[0]["reason_type"] == "duplicate"
    assert "duplicate of 'concept_b'" in removed[0]["reason_chip"]
