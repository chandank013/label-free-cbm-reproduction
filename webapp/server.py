import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Setup paths and environment
ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_LFCBM = ROOT / "original_lfcbm"
if str(ORIGINAL_LFCBM) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_LFCBM))

try:
    os.chdir(ORIGINAL_LFCBM)
except Exception:
    pass

from webapp.inference import InferenceEngine
from webapp.registry import KNOWN_BUDGETS, KNOWN_DATASETS, ModelRegistry
from webapp.results_reader import ResultsReader, PAPER_ACCURACY_PERCENT

app = FastAPI(title="Concept Lab - LF-CBM", version="1.0.0")

# Security: CORS for local interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry & Services
registry = ModelRegistry()
results_reader = ResultsReader(ROOT)
inference_engine = InferenceEngine(registry)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def validate_dataset(dataset: str) -> str:
    cleaned = dataset.strip().lower()
    if cleaned not in KNOWN_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset '{dataset}'. Must be one of: {', '.join(KNOWN_DATASETS)}",
        )
    return cleaned


def validate_budget(budget: int) -> int:
    if budget not in KNOWN_BUDGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid budget '{budget}'. Must be one of: {KNOWN_BUDGETS}",
        )
    return budget


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_file))


@app.get("/favicon.ico")
def serve_favicon():
    return Response(status_code=204)


@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "device": inference_engine.device,
        "cuda_available": inference_engine.device == "cuda",
    }


@app.get("/api/datasets")
def get_datasets():
    return {
        "datasets": registry.get_available_datasets(),
        "all_known": KNOWN_DATASETS,
        "budgets": KNOWN_BUDGETS,
    }


@app.get("/api/rescan")
def rescan_models():
    registry.rescan()
    return {
        "status": "rescanned",
        "datasets": registry.get_available_datasets(),
        "total_models": len(registry.get_all_models()),
    }


@app.get("/api/compare")
def get_compare_data(
    dataset: str = Query("cub", description="Dataset identifier"),
    budget: int = Query(50, description="Keep percentage budget"),
):
    ds = validate_dataset(dataset)
    b = validate_budget(budget)

    kpis = results_reader.get_kpis(ds, b)
    curves = results_reader.get_curve_series(ds)
    redundancy_comp = results_reader.get_redundancy_comparison(b)
    paper_vs_repro = results_reader.get_paper_vs_reproduced(ds)
    removed = results_reader.get_removed_concepts(ds, b)

    return {
        "dataset": ds,
        "budget": b,
        "kpis": kpis,
        "curves": curves,
        "redundancy_comparison": redundancy_comp,
        "paper_vs_reproduced": paper_vs_repro,
        "removed_concepts": removed,
    }


@app.get("/api/classes")
def get_classes(dataset: str = Query("cub")):
    ds = validate_dataset(dataset)
    classes = inference_engine.get_class_names(ds)
    return {
        "dataset": ds,
        "classes": classes,
        "count": len(classes),
    }


@app.get("/api/class_concepts")
def get_class_concepts(
    dataset: str = Query("cub"),
    class_idx: int = Query(0),
    budget: int = Query(50),
):
    ds = validate_dataset(dataset)
    b = validate_budget(budget)

    try:
        data = inference_engine.get_class_concepts(ds, class_idx, b)
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load class concepts: {e}")


@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    dataset: str = Form("cub"),
    budget: int = Form(50),
):
    ds = validate_dataset(dataset)
    b = validate_budget(budget)

    # Check available models
    models = registry.get_dataset_models(ds)
    if not models.get("original") or not models.get(str(b)):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{ds}' does not have a complete model pair at budget {b}%.",
        )

    # Read bytes
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not content:
        raise HTTPException(status_code=400, detail="Empty upload.")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds 10 MB limit.")

    try:
        result = inference_engine.predict(ds, b, content)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/api/about_info")
def get_about_info():
    """Return model inventory and metadata for About page."""
    models_info = []
    for m in registry.get_all_models():
        models_info.append({
            "dataset": m.dataset,
            "folder_name": m.folder_name,
            "method": m.method,
            "budget": m.budget,
            "seed": m.seed,
            "is_tagged": m.is_tagged,
            "label": m.label,
        })

    active_pairs = {}
    for ds in KNOWN_DATASETS:
        ds_models = registry.get_dataset_models(ds)
        active_pairs[ds] = {
            "original": ds_models["original"].folder_name if ds_models["original"] else None,
            "75": ds_models["75"].folder_name if ds_models["75"] else None,
            "50": ds_models["50"].folder_name if ds_models["50"] else None,
            "25": ds_models["25"].folder_name if ds_models["25"] else None,
        }

    return {
        "active_pairs": active_pairs,
        "total_valid_models": len(models_info),
        "models": models_info,
        "device": inference_engine.device,
        "paper_numbers": PAPER_ACCURACY_PERCENT,
    }
