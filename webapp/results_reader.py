import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PAPER_ACCURACY_PERCENT = {
    "cifar10": 86.37,
    "cifar100": 65.27,
    "cub": 74.59,
    "places365": 43.71,
    "imagenet": 71.98,
}

ALL_DATASETS = ["cifar10", "cifar100", "cub", "places365", "imagenet"]


def normalize_acc(val: Optional[float]) -> Optional[float]:
    """Defensively normalize accuracy to a fraction in [0, 1].
    
    If val > 1.5, it was stored as a percentage (e.g. 86.4) and is divided by 100.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if f < 0:
            return None
        if f > 1.5:
            return f / 100.0
        return f
    except (ValueError, TypeError):
        return None


def mean_std(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, sample_std, count). If count <= 1, std is 0.0."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    n = len(valid)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(valid) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in valid) / (n - 1)
    return m, math.sqrt(var), n


class ResultsReader:
    def __init__(self, workspace_root: Optional[Path] = None):
        if workspace_root is None:
            self.root = Path(__file__).resolve().parent.parent
        else:
            self.root = Path(workspace_root)
        self.results_dir = self.root / "results"
        self.phase2_dir = self.results_dir / "phase2"

    def read_phase1_reproduced(self, dataset: str) -> Optional[float]:
        """Read reproduced accuracy for Phase 1.
        
        Checks results/lfcbm_<dataset>_seed0.json first.
        If null, falls back to saved model metrics.txt.
        """
        json_path = self.results_dir / f"lfcbm_{dataset}_seed0.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                acc = data.get("reproduced_accuracy")
                if acc is not None:
                    return normalize_acc(acc)
            except Exception:
                pass

        # Fallback to saved_models baseline metrics.txt
        saved_models_dir = self.root / "original_lfcbm" / "saved_models"
        if saved_models_dir.exists():
            for d in saved_models_dir.iterdir():
                if d.is_dir() and d.name.startswith(f"{dataset}_cbm") and ("baseline" in d.name or "s0" in d.name):
                    metrics_path = d / "metrics.txt"
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, "r", encoding="utf-8") as f:
                                mdata = json.load(f)
                            m_acc = mdata.get("metrics", {}).get("acc_val")
                            if m_acc is not None and m_acc != -1:
                                return normalize_acc(m_acc)
                        except Exception:
                            pass
        return None

    def get_phase2_runs(self, dataset: str) -> List[Dict[str, Any]]:
        """Load all Phase 2 result JSON files for a dataset."""
        ds_dir = self.phase2_dir / dataset
        if not ds_dir.exists():
            return []

        runs = []
        for p in ds_dir.glob("*.json"):
            if not p.is_file():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                summary = data.get("summary", {})
                if not summary:
                    continue
                # Normalize accuracy
                if "acc_val" in summary:
                    summary["acc_val"] = normalize_acc(summary["acc_val"])
                runs.append({
                    "filename": p.name,
                    "summary": summary,
                    "removed_by_filter": data.get("removed_by_filter", []),
                })
            except Exception:
                continue
        return runs

    def get_kpis(self, dataset: str, budget: int) -> Dict[str, Any]:
        """Compute KPI cards comparing Original vs Filtered."""
        runs = self.get_phase2_runs(dataset)
        
        # Original: method == 'baseline' (or keep_frac == 1.0)
        orig_runs = [
            r["summary"] for r in runs
            if r["summary"].get("method") == "baseline" or r["summary"].get("keep_frac") == 1.0
        ]
        # Filtered: method == 'greedy' and budget matches
        filtered_runs = [
            r["summary"] for r in runs
            if r["summary"].get("method") == "greedy" and (
                int(round(r["summary"].get("keep_frac", 0) * 100)) == budget
                or f"_k{budget}_" in r["summary"].get("tag", "")
            )
        ]

        def compute_kpi(metric_key: str, higher_is_better: bool) -> Dict[str, Any]:
            orig_vals = [r[metric_key] for r in orig_runs if metric_key in r and r[metric_key] is not None]
            filt_vals = [r[metric_key] for r in filtered_runs if metric_key in r and r[metric_key] is not None]

            if not orig_vals and not filt_vals:
                return {
                    "original": None,
                    "filtered": None,
                    "delta": None,
                    "pct_change": None,
                    "status": "not_run",
                    "badge_color": "grey",
                    "badge_text": "Not run",
                }

            orig_m, orig_s, orig_n = mean_std(orig_vals) if orig_vals else (None, 0.0, 0)
            filt_m, filt_s, filt_n = mean_std(filt_vals) if filt_vals else (None, 0.0, 0)

            if orig_m is None or filt_m is None:
                return {
                    "original": orig_m,
                    "filtered": filt_m,
                    "delta": None,
                    "pct_change": None,
                    "status": "partial",
                    "badge_color": "grey",
                    "badge_text": "Partial data",
                    "orig_std": orig_s,
                    "orig_seeds": orig_n,
                    "filt_std": filt_s,
                    "filt_seeds": filt_n,
                }

            delta = filt_m - orig_m
            pct_change = (delta / orig_m * 100.0) if orig_m != 0 else 0.0

            # Delta badge rules
            abs_delta = abs(delta)
            if orig_n > 1 and abs_delta <= orig_s:
                badge_color = "grey"
                badge_text = "within seed noise"
            else:
                is_improvement = (delta > 0) if higher_is_better else (delta < 0)
                if abs_delta < 1e-6:
                    badge_color = "grey"
                    badge_text = "no change"
                elif is_improvement:
                    badge_color = "green"
                    badge_text = f"{'+' if delta > 0 else ''}{delta:.3f} ({'+' if pct_change > 0 else ''}{pct_change:.1f}%)"
                else:
                    badge_color = "red"
                    badge_text = f"{'+' if delta > 0 else ''}{delta:.3f} ({'+' if pct_change > 0 else ''}{pct_change:.1f}%)"

            seed_note = "1 seed - no error bars" if (orig_n <= 1 or filt_n <= 1) else f"{orig_n} seeds"

            return {
                "original": orig_m,
                "filtered": filt_m,
                "delta": delta,
                "pct_change": pct_change,
                "orig_std": orig_s,
                "orig_seeds": orig_n,
                "filt_std": filt_s,
                "filt_seeds": filt_n,
                "status": "ok",
                "badge_color": badge_color,
                "badge_text": badge_text,
                "seed_note": seed_note,
            }

        return {
            "accuracy": compute_kpi("acc_val", higher_is_better=True),
            "n_concepts": compute_kpi("n_concepts", higher_is_better=False),
            "redundancy": compute_kpi("redundancy_nn_abs_corr", higher_is_better=False),
            "relevance": compute_kpi("relevance_mean", higher_is_better=True),
        }

    def get_curve_series(self, dataset: str) -> Dict[str, Any]:
        """Return curve data for Accuracy vs Number of Concepts.
        
        Series:
        - Ours (Filtered): greedy
        - Top-k by CLIP activation: clip_topk
        - Random: random
        - Original: baseline (plotted as star)
        """
        runs = self.get_phase2_runs(dataset)
        if not runs:
            return {"status": "not_run", "series": {}}

        # Methods to extract
        methods = {
            "greedy": "Ours (Filtered)",
            "clip_topk": "Top-k by CLIP activation",
            "random": "Random",
        }

        series_data: Dict[str, List[Dict[str, Any]]] = {
            "greedy": [],
            "clip_topk": [],
            "random": [],
        }

        # Original point
        orig_runs = [r["summary"] for r in runs if r["summary"].get("method") == "baseline"]
        orig_point = None
        if orig_runs:
            acc_m, acc_s, _ = mean_std([r["acc_val"] for r in orig_runs if "acc_val" in r])
            k_m, _, _ = mean_std([r["n_concepts"] for r in orig_runs if "n_concepts" in r])
            orig_point = {
                "n_concepts": round(k_m),
                "accuracy": acc_m,
                "std": acc_s,
            }

        for method_key in methods.keys():
            # Group by budget / keep_frac
            method_runs = [r["summary"] for r in runs if r["summary"].get("method") == method_key]
            # Group by keep_frac
            by_frac: Dict[float, List[Dict]] = {}
            for r in method_runs:
                frac = r.get("keep_frac", 0.0)
                by_frac.setdefault(frac, []).append(r)

            points = []
            for frac in sorted(by_frac.keys()):
                group = by_frac[frac]
                acc_m, acc_s, n_s = mean_std([item["acc_val"] for item in group if "acc_val" in item])
                k_m, _, _ = mean_std([item["n_concepts"] for item in group if "n_concepts" in item])
                points.append({
                    "keep_frac": frac,
                    "budget": int(round(frac * 100)),
                    "n_concepts": round(k_m),
                    "accuracy": acc_m,
                    "std": acc_s,
                    "n_seeds": n_s,
                })
            points.sort(key=lambda p: p["n_concepts"])
            series_data[method_key] = points

        return {
            "status": "ok",
            "series": series_data,
            "original_point": orig_point,
        }

    def get_redundancy_comparison(self, budget: int) -> List[Dict[str, Any]]:
        """Return redundancy comparison (Original vs Filtered) for ALL 5 datasets."""
        results = []
        for ds in ALL_DATASETS:
            runs = self.get_phase2_runs(ds)
            if not runs:
                results.append({
                    "dataset": ds,
                    "display_name": format_dataset_name(ds),
                    "status": "not_run",
                    "original": None,
                    "filtered": None,
                })
                continue

            orig_runs = [r["summary"] for r in runs if r["summary"].get("method") == "baseline"]
            filt_runs = [
                r["summary"] for r in runs
                if r["summary"].get("method") == "greedy" and (
                    int(round(r["summary"].get("keep_frac", 0) * 100)) == budget
                    or f"_k{budget}_" in r["summary"].get("tag", "")
                )
            ]

            orig_vals = [r["redundancy_nn_abs_corr"] for r in orig_runs if "redundancy_nn_abs_corr" in r]
            filt_vals = [r["redundancy_nn_abs_corr"] for r in filt_runs if "redundancy_nn_abs_corr" in r]

            orig_m, _, _ = mean_std(orig_vals) if orig_vals else (None, 0.0, 0)
            filt_m, _, _ = mean_std(filt_vals) if filt_vals else (None, 0.0, 0)

            if orig_m is None and filt_m is None:
                results.append({
                    "dataset": ds,
                    "display_name": format_dataset_name(ds),
                    "status": "not_run",
                    "original": None,
                    "filtered": None,
                })
            else:
                results.append({
                    "dataset": ds,
                    "display_name": format_dataset_name(ds),
                    "status": "ok",
                    "original": orig_m,
                    "filtered": filt_m,
                })
        return results

    def get_paper_vs_reproduced(self, selected_dataset: str) -> List[Dict[str, Any]]:
        """Return paper vs reproduced table rows for all 5 datasets."""
        rows = []
        for ds in ALL_DATASETS:
            paper_pct = PAPER_ACCURACY_PERCENT.get(ds)
            reproduced_frac = self.read_phase1_reproduced(ds)
            
            if reproduced_frac is None:
                rows.append({
                    "dataset": ds,
                    "display_name": format_dataset_name(ds),
                    "paper": paper_pct / 100.0 if paper_pct is not None else None,
                    "paper_pct": paper_pct,
                    "reproduced": None,
                    "reproduced_pct": None,
                    "diff": None,
                    "diff_pct": None,
                    "status": "not_run",
                    "is_selected": ds == selected_dataset,
                })
            else:
                paper_frac = paper_pct / 100.0
                reproduced_pct = reproduced_frac * 100.0
                diff_frac = reproduced_frac - paper_frac
                diff_pct = reproduced_pct - paper_pct
                rows.append({
                    "dataset": ds,
                    "display_name": format_dataset_name(ds),
                    "paper": paper_frac,
                    "paper_pct": paper_pct,
                    "reproduced": reproduced_frac,
                    "reproduced_pct": reproduced_pct,
                    "diff": diff_frac,
                    "diff_pct": diff_pct,
                    "status": "ok",
                    "is_selected": ds == selected_dataset,
                })
        return rows

    def get_removed_concepts(self, dataset: str, budget: int) -> List[Dict[str, Any]]:
        """Return removed concepts table with reason chips and nearest kept concept."""
        runs = self.get_phase2_runs(dataset)
        # Find greedy run for given budget, lowest seed
        matching_runs = [
            r for r in runs
            if r["summary"].get("method") == "greedy" and (
                int(round(r["summary"].get("keep_frac", 0) * 100)) == budget
                or f"_k{budget}_" in r["summary"].get("tag", "")
            )
        ]
        if not matching_runs:
            return []

        # Sort by seed
        matching_runs.sort(key=lambda r: r["summary"].get("seed", 999))
        best_run = matching_runs[0]
        raw_removed = best_run.get("removed_by_filter", [])

        results = []
        for i, item in enumerate(raw_removed, start=1):
            concept = item.get("concept", "")
            relevance = item.get("relevance", 0.0)
            nearest = item.get("nearest_kept") or "—"
            corr = item.get("corr_with_nearest_kept")

            # Contract Rule:
            # "duplicate of '<nearest_kept>' (r=<value>)" in an amber chip when corr_with_nearest_kept >= 0.80,
            # otherwise a purple "low relevance (eta2=<value>)" chip.
            is_dup = (corr is not None) and (corr >= 0.80)
            if is_dup:
                reason_type = "duplicate"
                reason_chip = f"duplicate of '{nearest}' (r={corr:.2f})"
                example = nearest
            else:
                reason_type = "low_relevance"
                reason_chip = f"low relevance (η²={relevance:.2f})"
                example = "—"

            results.append({
                "index": i,
                "concept": concept,
                "reason_type": reason_type,
                "reason_chip": reason_chip,
                "example": example,
                "relevance": relevance,
                "corr": corr,
                "nearest_kept": nearest,
            })
        return results


def format_dataset_name(ds: str) -> str:
    mapping = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "cub": "CUB-200",
        "places365": "Places365",
        "imagenet": "ImageNet",
    }
    return mapping.get(ds, ds.upper())
