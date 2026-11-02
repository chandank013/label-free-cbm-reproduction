import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REQUIRED_FILES = [
    "args.txt",
    "metrics.txt",
    "concepts.txt",
    "W_c.pt",
    "W_g.pt",
    "b_g.pt",
    "proj_mean.pt",
    "proj_std.pt",
]

KNOWN_DATASETS = ["cifar10", "cifar100", "cub", "places365", "imagenet"]
KNOWN_BUDGETS = [75, 50, 25]

# Regex to match dataset and base name
# Examples:
#   cifar10_cbm_2026_11_01_15_53_31_greedy_k50_s0
#   cifar100_cbm_2026_10_24_18_38
#   cub_lf_cbm
FOLDER_REGEX = re.compile(
    r"^(cifar100|cifar10|cub|places365|imagenet)_(?:cbm_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}(?:_\d{2})?)|lf_cbm)(.*)$"
)

TAG_REGEX = re.compile(r"^_(.+)_k(\d+)_s(\d+)$")


@dataclass
class ModelInfo:
    dataset: str
    folder_name: str
    path: Path
    timestamp: str  # e.g. "2026_11_01_15_53_31" or ""
    method: str  # "baseline", "greedy", "random", "clip_topk", "untagged", etc.
    budget: Optional[int]  # 100, 75, 50, 25 or None
    seed: Optional[int]  # 0, 1, 2 or None
    is_tagged: bool
    label: str  # Display label


class ModelRegistry:
    def __init__(self, saved_models_dir: Optional[Path] = None):
        if saved_models_dir is None:
            # Workspace root / original_lfcbm / saved_models
            root = Path(__file__).resolve().parent.parent
            saved_models_dir = root / "original_lfcbm" / "saved_models"
        self.saved_models_dir = Path(saved_models_dir)
        self._all_models: List[ModelInfo] = []
        self._dataset_map: Dict[str, Dict[str, Optional[ModelInfo]]] = {}
        self.rescan()

    @staticmethod
    def is_folder_complete(folder: Path) -> bool:
        if not folder.is_dir():
            return False
        for req in REQUIRED_FILES:
            if not (folder / req).is_file():
                return False
        return True

    @classmethod
    def parse_folder_name(cls, folder_path: Path) -> Optional[ModelInfo]:
        folder_name = folder_path.name
        match = FOLDER_REGEX.match(folder_name)
        if not match:
            return None

        dataset, timestamp, rest = match.groups()
        timestamp = timestamp or ""

        if rest:
            tag_match = TAG_REGEX.match(rest)
            if tag_match:
                method, budget_str, seed_str = tag_match.groups()
                budget = int(budget_str)
                seed = int(seed_str)
                return ModelInfo(
                    dataset=dataset,
                    folder_name=folder_name,
                    path=folder_path,
                    timestamp=timestamp,
                    method=method,
                    budget=budget,
                    seed=seed,
                    is_tagged=True,
                    label=f"_{method}_k{budget}_s{seed}",
                )
            # If rest doesn't match standard tag format, treat as custom/untagged with rest
            return ModelInfo(
                dataset=dataset,
                folder_name=folder_name,
                path=folder_path,
                timestamp=timestamp,
                method=rest.lstrip("_"),
                budget=None,
                seed=None,
                is_tagged=False,
                label=folder_name,
            )
        else:
            # No tag suffix
            return ModelInfo(
                dataset=dataset,
                folder_name=folder_name,
                path=folder_path,
                timestamp=timestamp,
                method="untagged",
                budget=None,
                seed=None,
                is_tagged=False,
                label="Phase 1 (untagged)",
            )

    def rescan(self) -> None:
        """Scan directory and recompute model pairings."""
        self._all_models.clear()
        self._dataset_map.clear()

        if not self.saved_models_dir.exists():
            return

        for child in sorted(self.saved_models_dir.iterdir()):
            if not child.is_dir():
                continue
            if not self.is_folder_complete(child):
                continue

            info = self.parse_folder_name(child)
            if info:
                self._all_models.append(info)

        # Build active model exposure for each known dataset
        for ds in KNOWN_DATASETS:
            ds_models = [m for m in self._all_models if m.dataset == ds]
            self._dataset_map[ds] = self._select_models_for_dataset(ds_models)

    @staticmethod
    def _select_models_for_dataset(models: List[ModelInfo]) -> Dict[str, Optional[ModelInfo]]:
        """
        Selection contract:
        - original:
            1) tagged baseline_k100: lowest seed, newest timestamp on duplicate.
            2) fallback: untagged model, newest timestamp wins (or mtime).
        - filtered per budget (75, 50, 25):
            greedy_k<budget>: lowest seed, newest timestamp on duplicate.
        """
        result: Dict[str, Optional[ModelInfo]] = {
            "original": None,
            "75": None,
            "50": None,
            "25": None,
        }

        # 1. Original model
        baseline_candidates = [
            m for m in models
            if m.method == "baseline" and m.budget == 100 and m.seed is not None
        ]
        if baseline_candidates:
            sorted_baselines = sort_model_candidates(baseline_candidates)
            best_baseline = sorted_baselines[0]
            best_baseline.label = f"baseline_k100_s{best_baseline.seed}"
            result["original"] = best_baseline
        else:
            # Fallback to untagged
            untagged_candidates = [m for m in models if not m.is_tagged]
            if untagged_candidates:
                sorted_untagged = sort_model_candidates(untagged_candidates)
                best_untagged = sorted_untagged[0]
                best_untagged.label = "Phase 1 (untagged)"
                result["original"] = best_untagged

        # 2. Filtered models per budget
        for budget in KNOWN_BUDGETS:
            greedy_candidates = [
                m for m in models
                if m.method == "greedy" and m.budget == budget and m.seed is not None
            ]
            if greedy_candidates:
                sorted_greedy = sort_model_candidates(greedy_candidates)
                result[str(budget)] = sorted_greedy[0]

        return result

    def get_dataset_models(self, dataset: str) -> Dict[str, Optional[ModelInfo]]:
        return self._dataset_map.get(dataset, {"original": None, "75": None, "50": None, "25": None})

    def get_all_models(self) -> List[ModelInfo]:
        return list(self._all_models)

    def get_available_datasets(self) -> List[Dict]:
        """Return dataset info for UI dataset picker."""
        output = []
        for ds in KNOWN_DATASETS:
            models = self.get_dataset_models(ds)
            orig = models.get("original")
            available_budgets = [b for b in KNOWN_BUDGETS if models.get(str(b)) is not None]
            has_pair = orig is not None and len(available_budgets) > 0
            output.append({
                "id": ds,
                "name": format_dataset_name(ds),
                "has_original": orig is not None,
                "original_label": orig.label if orig else None,
                "available_budgets": available_budgets,
                "has_pair": has_pair,
            })
        return output


def sort_model_candidates(candidates: List[ModelInfo]) -> List[ModelInfo]:
    """Sort candidates: lowest seed first; newest timestamp wins on ties."""
    if not candidates:
        return []
    # 1. Sort by timestamp/mtime descending (newest first)
    by_time = sorted(
        candidates,
        key=lambda m: (
            m.timestamp or "",
            m.path.stat().st_mtime if m.path.exists() else 0,
        ),
        reverse=True,
    )
    # 2. Stable sort by seed ascending (lowest seed first)
    return sorted(by_time, key=lambda m: m.seed if m.seed is not None else 999999)


def format_dataset_name(ds: str) -> str:
    mapping = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "cub": "CUB-200 birds",
        "places365": "Places365",
        "imagenet": "ImageNet",
    }
    return mapping.get(ds, ds.upper())
