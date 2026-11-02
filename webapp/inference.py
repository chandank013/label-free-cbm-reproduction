import io
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageOps

# Ensure original_lfcbm is on sys.path and is current working dir for relative assets
ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_LFCBM = ROOT / "original_lfcbm"
if str(ORIGINAL_LFCBM) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_LFCBM))

try:
    os.chdir(ORIGINAL_LFCBM)
except Exception:
    pass

import cbm  # type: ignore
import data_utils  # type: ignore

from webapp.registry import ModelInfo, ModelRegistry

STOP_WORDS = {
    "a", "an", "the", "with", "of", "and", "in", "on", "for", "is", "at", "by",
    "from", "to", "or", "as", "that", "this", "it", "its", "are", "be", "has", "have"
}


def clean_words(text: str) -> set:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def word_jaccard(s1: str, s2: str) -> float:
    w1 = clean_words(s1)
    w2 = clean_words(s2)
    if not w1 or not w2:
        return 0.0
    union = len(w1 | w2)
    return len(w1 & w2) / union if union > 0 else 0.0


class ModelPairSession:
    def __init__(self, dataset: str, budget: int, orig_info: ModelInfo, filt_info: ModelInfo, device: str):
        self.dataset = dataset
        self.budget = budget
        self.orig_info = orig_info
        self.filt_info = filt_info
        self.device = device

        # Load models using authors' loader
        self.orig_model = cbm.load_cbm(str(orig_info.path), device)
        self.filt_model = cbm.load_cbm(str(filt_info.path), device)

        self.orig_model.eval()
        self.filt_model.eval()

        # Load concepts
        self.orig_concepts = self._load_concepts(orig_info.path)
        self.filt_concepts = self._load_concepts(filt_info.path)

        # Load preprocessors
        orig_args = json.load(open(orig_info.path / "args.txt", "r", encoding="utf-8"))
        filt_args = json.load(open(filt_info.path / "args.txt", "r", encoding="utf-8"))

        _, self.orig_preprocess = data_utils.get_target_model(orig_args["backbone"], device)
        _, self.filt_preprocess = data_utils.get_target_model(filt_args["backbone"], device)

        # Load redundancy metrics from phase2_report or summary if available
        self.orig_redundancy = self._load_redundancy(orig_info.path)
        self.filt_redundancy = self._load_redundancy(filt_info.path)

    @staticmethod
    def _load_concepts(path: Path) -> List[str]:
        concepts_file = path / "concepts.txt"
        if concepts_file.exists():
            with open(concepts_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    @staticmethod
    def _load_redundancy(path: Path) -> Optional[float]:
        # Check phase2_report.json
        p2_report = path / "phase2_report.json"
        if p2_report.exists():
            try:
                data = json.load(open(p2_report, "r", encoding="utf-8"))
                return data.get("summary", {}).get("redundancy_nn_abs_corr")
            except Exception:
                pass
        return None


class InferenceEngine:
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._current_session: Optional[ModelPairSession] = None
        self._class_names_cache: Dict[str, List[str]] = {}

    def get_class_names(self, dataset: str) -> List[str]:
        if dataset in self._class_names_cache:
            return self._class_names_cache[dataset]
        if dataset not in data_utils.LABEL_FILES:
            return []
        cls_file = ORIGINAL_LFCBM / data_utils.LABEL_FILES[dataset]
        if not cls_file.exists():
            cls_file = ROOT / "original_lfcbm" / data_utils.LABEL_FILES[dataset]
        if not cls_file.exists():
            return []
        with open(cls_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        self._class_names_cache[dataset] = classes
        return classes

    def _get_or_load_session(self, dataset: str, budget: int) -> ModelPairSession:
        """Lazy load model pair, evicting any previous pair."""
        models = self.registry.get_dataset_models(dataset)
        orig_info = models.get("original")
        filt_info = models.get(str(budget))

        if orig_info is None:
            raise ValueError(f"No original model found for dataset '{dataset}'")
        if filt_info is None:
            raise ValueError(f"No filtered model found for dataset '{dataset}' at budget {budget}%")

        if (
            self._current_session is not None
            and self._current_session.dataset == dataset
            and self._current_session.budget == budget
            and self._current_session.orig_info.path == orig_info.path
            and self._current_session.filt_info.path == filt_info.path
        ):
            return self._current_session

        # Evict previous
        if self._current_session is not None:
            del self._current_session
            self._current_session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load new session
        session = ModelPairSession(dataset, budget, orig_info, filt_info, self.device)
        self._current_session = session
        return session

    def process_image(self, file_bytes: bytes) -> Image.Image:
        """Validate upload safety: max 10MB, valid JPEG/PNG/WEBP, EXIF rotated, RGB."""
        if len(file_bytes) > 10 * 1024 * 1024:
            raise ValueError("File size exceeds 10 MB limit.")

        try:
            img = Image.open(io.BytesIO(file_bytes))
            fmt = (img.format or "").upper()
            if fmt not in ("JPEG", "PNG", "WEBP", "JPG"):
                raise ValueError(f"Unsupported image format: {fmt}. Only JPEG, PNG, WEBP allowed.")
            # Load pixel data completely into memory
            img.load()
            # Apply EXIF rotation safely
            img = ImageOps.exif_transpose(img)
            # Convert to RGB
            img = img.convert("RGB")
            return img
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable image file: {e}")

    def predict(self, dataset: str, budget: int, file_bytes: bytes) -> Dict[str, Any]:
        """Run side-by-side inference for Original and Filtered models."""
        image = self.process_image(file_bytes)
        class_names = self.get_class_names(dataset)

        with self._lock:
            session = self._get_or_load_session(dataset, budget)

            # Preprocess
            tensor_orig = session.orig_preprocess(image).unsqueeze(0).to(self.device)
            tensor_filt = session.filt_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits_orig, proj_c_orig = session.orig_model(tensor_orig)
                logits_filt, proj_c_filt = session.filt_model(tensor_filt)

                probs_orig = torch.softmax(logits_orig, dim=-1)[0]
                probs_filt = torch.softmax(logits_filt, dim=-1)[0]

            # Parse predictions
            pred_orig = self._format_prediction(
                session.orig_model,
                session.orig_concepts,
                class_names,
                probs_orig,
                proj_c_orig[0],
                session.orig_redundancy,
                card_type="original",
            )
            pred_filt = self._format_prediction(
                session.filt_model,
                session.filt_concepts,
                class_names,
                probs_filt,
                proj_c_filt[0],
                session.filt_redundancy,
                card_type="filtered",
            )

        models_agree = (pred_orig["top_class_index"] == pred_filt["top_class_index"])

        return {
            "dataset": dataset,
            "budget": budget,
            "models_agree": models_agree,
            "agreement_badge": {
                "text": "Both models agree" if models_agree else "Models disagree",
                "color": "green" if models_agree else "amber",
            },
            "original": pred_orig,
            "filtered": pred_filt,
            "all_classes": class_names,
            "device": self.device,
        }

    def _format_prediction(
        self,
        model: Any,
        concepts: List[str],
        class_names: List[str],
        probs: torch.Tensor,
        proj_c: torch.Tensor,
        redundancy_metric: Optional[float],
        card_type: str,
    ) -> Dict[str, Any]:
        num_classes = len(probs)
        topk_k = min(3, num_classes)
        topk_probs, topk_indices = torch.topk(probs, k=topk_k)

        top3 = []
        for prob_val, idx in zip(topk_probs, topk_indices):
            c_idx = idx.item()
            name = class_names[c_idx] if c_idx < len(class_names) else f"Class {c_idx}"
            top3.append({
                "class_index": c_idx,
                "class_name": name,
                "confidence": float(prob_val.item()),
            })

        pred_idx = topk_indices[0].item()
        pred_class_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
        confidence = float(topk_probs[0].item())

        # Final layer weights W_g: [num_classes, num_concepts]
        W_g = model.final.weight.data
        class_weights = W_g[pred_idx]  # [num_concepts]

        # Per-concept contribution to predicted class = proj_c * W_g[pred_idx]
        contributions = proj_c * class_weights  # [num_concepts]

        # Top 6 positive contributors
        pos_mask = contributions > 0
        pos_indices = torch.nonzero(pos_mask).squeeze(-1)

        concept_list: List[Dict[str, Any]] = []
        if pos_indices.numel() > 0:
            pos_contribs = contributions[pos_indices]
            sorted_order = torch.argsort(pos_contribs, descending=True)
            top_k_indices = pos_indices[sorted_order][:6]

            for rank, c_i in enumerate(top_k_indices):
                idx = c_i.item()
                raw_val = float(contributions[idx].item())
                concept_text = concepts[idx] if idx < len(concepts) else f"Concept {idx}"
                concept_list.append({
                    "concept_index": idx,
                    "concept": concept_text,
                    "contribution": raw_val,
                    "activation": float(proj_c[idx].item()),
                    "weight": float(class_weights[idx].item()),
                    "is_duplicate": False,
                })

        # Try-it "duplicate" tags: word Jaccard >= 0.5 after stop-word removal among displayed top concepts
        for i in range(len(concept_list)):
            c_text_i = concept_list[i]["concept"]
            for j in range(i):
                c_text_j = concept_list[j]["concept"]
                if word_jaccard(c_text_i, c_text_j) >= 0.5:
                    concept_list[i]["is_duplicate"] = True
                    concept_list[i]["duplicate_of"] = c_text_j
                    break

        # Max contribution for relative bar scaling in UI
        max_contrib = max([c["contribution"] for c in concept_list], default=1.0)
        for c in concept_list:
            c["bar_pct"] = (c["contribution"] / max_contrib * 100.0) if max_contrib > 0 else 0.0

        # NEC: non-zero weights per class (|w| > 1e-5)
        nz_pred = (class_weights.abs() > 1e-5).sum().item()
        nz_all = (W_g.abs() > 1e-5).sum().item()
        nec_avg = nz_all / num_classes if num_classes > 0 else 0.0

        num_concepts = model.proj_layer.weight.shape[0]

        return {
            "card_type": card_type,
            "top_class_index": pred_idx,
            "predicted_class": pred_class_name,
            "confidence": confidence,
            "is_low_confidence": confidence < 0.5,
            "top3": top3,
            "top_concepts": concept_list,
            "stats": {
                "n_concepts": num_concepts,
                "concepts_per_class": round(nec_avg, 1),
                "nec_predicted_class": nz_pred,
                "redundancy": round(redundancy_metric, 2) if redundancy_metric is not None else None,
            },
        }

    def get_class_concepts(self, dataset: str, class_idx: int, budget: int) -> Dict[str, Any]:
        """Return top-weighted concepts for a given class from both models."""
        class_names = self.get_class_names(dataset)
        if class_idx < 0 or class_idx >= len(class_names):
            raise ValueError(f"Class index {class_idx} out of range for dataset '{dataset}'")

        class_name = class_names[class_idx]

        with self._lock:
            session = self._get_or_load_session(dataset, budget)

            def extract_class_top(model: Any, concepts: List[str]) -> List[Dict[str, Any]]:
                W_g = model.final.weight.data
                weights = W_g[class_idx]
                topk_k = min(15, weights.shape[0])
                top_vals, top_idx = torch.topk(weights, k=topk_k)
                out = []
                for val, idx in zip(top_vals, top_idx):
                    i = idx.item()
                    c_name = concepts[i] if i < len(concepts) else f"Concept {i}"
                    out.append({
                        "concept": c_name,
                        "weight": float(val.item()),
                    })
                return out

            orig_top = extract_class_top(session.orig_model, session.orig_concepts)
            filt_top = extract_class_top(session.filt_model, session.filt_concepts)

        return {
            "dataset": dataset,
            "class_index": class_idx,
            "class_name": class_name,
            "budget": budget,
            "original_concepts": orig_top,
            "filtered_concepts": filt_top,
        }
