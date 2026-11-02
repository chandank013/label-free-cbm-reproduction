import io
import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Ensure original_lfcbm is on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "original_lfcbm"))
os.chdir(ROOT / "original_lfcbm")

import cbm
import data_utils

from webapp.server import app, registry, inference_engine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_real_model_smoke_and_cross_check(client):
    """Test /api/predict on all datasets with a valid model pair, reporting time and cross-checking with cbm.load_cbm."""
    available = registry.get_available_datasets()
    paired_datasets = [d["id"] for d in available if d["has_pair"]]
    
    print(f"\nDatasets with valid model pair: {paired_datasets}")
    assert len(paired_datasets) > 0, "No datasets with valid model pair found!"

    # Create a test image
    img = Image.new("RGB", (224, 224), color=(200, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    for ds in paired_datasets:
        budget = 50
        print(f"\n--- Testing Dataset: {ds} (Budget {budget}%) ---")
        
        # 1. API Call
        start_t = time.perf_counter()
        res = client.post(
            "/api/predict",
            data={"dataset": ds, "budget": str(budget)},
            files={"file": ("sample.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        elapsed = time.perf_counter() - start_t
        assert res.status_code == 200, f"API failed for {ds}: {res.text}"
        data = res.json()

        api_orig = data["original"]
        api_filt = data["filtered"]
        print(f"API Time: {elapsed:.3f}s")
        print(f"API Original -> Class: '{api_orig['predicted_class']}', Conf: {api_orig['confidence']:.4f}")
        print(f"API Filtered -> Class: '{api_filt['predicted_class']}', Conf: {api_filt['confidence']:.4f}")
        print(f"Models agree: {data['models_agree']}")
        print(f"Top 3 original concepts: {[c['concept'] for c in api_orig['top_concepts'][:3]]}")

        # 2. Cross-check directly with cbm.load_cbm
        models = registry.get_dataset_models(ds)
        orig_info = models["original"]
        assert orig_info is not None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        direct_model = cbm.load_cbm(str(orig_info.path), device)
        direct_model.eval()

        args = json.load(open(orig_info.path / "args.txt", "r", encoding="utf-8"))
        _, direct_preprocess = data_utils.get_target_model(args["backbone"], device)

        with torch.no_grad():
            tensor = direct_preprocess(img).unsqueeze(0).to(device)
            logits, _ = direct_model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            top_prob, top_idx = torch.topk(probs, k=1)
            direct_conf = float(top_prob.item())
            direct_class_idx = int(top_idx.item())

        classes = inference_engine.get_class_names(ds)
        direct_class_name = classes[direct_class_idx]

        print(f"Direct Model -> Class: '{direct_class_name}', Conf: {direct_conf:.4f}")

        # Assert top-1 class matches
        assert api_orig["predicted_class"] == direct_class_name, (
            f"Class mismatch for {ds}: API '{api_orig['predicted_class']}' vs Direct '{direct_class_name}'"
        )

        # Assert confidence agrees within tolerance 1e-4
        assert abs(api_orig["confidence"] - direct_conf) < 1e-4, (
            f"Confidence mismatch for {ds}: API {api_orig['confidence']} vs Direct {direct_conf}"
        )
        print(f"Cross-check PASSED for {ds} (tolerance < 1e-4)")
