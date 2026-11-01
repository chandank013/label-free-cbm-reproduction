"""
Phase 1 -- reproducibility wrapper around the authors' LF-CBM training code.

It does NOT reimplement the method: it fixes the random seeds, applies the
runtime compatibility layer (compat.py), then calls the authors' own
train_cbm.train_cbm_and_save() / train_standard.train_standard() and logs the
run to results/<model>_<dataset>_seed<seed>.json.

    python run_experiment.py --dataset cifar10   --model lfcbm --seed 0 --batch_size 64
    python run_experiment.py --dataset cub       --model lfcbm --seed 0 --batch_size 64
    python run_experiment.py --dataset places365 --model lfcbm --seed 0 --batch_size 32
    python run_experiment.py --dataset imagenet  --model lfcbm --seed 0 --batch_size 32 --imagenet_root D:/datasets/ImageNet
    (repeat with --model standard_sparse for the baseline rows)

Hyper-parameters are taken verbatim from original_lfcbm/training_commands.txt
(see docs/DATASET_COMMANDS.md).
"""
import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import compat  # noqa: E402  (must be imported before torch: sets allocator env on non-Windows)

import numpy as np  # noqa: E402
import torch  # noqa: E402

LFCBM_CONFIGS = {
    "cifar10":   dict(dataset="cifar10",   concept_set="data/concept_sets/cifar10_filtered.txt"),
    "cifar100":  dict(dataset="cifar100",  concept_set="data/concept_sets/cifar100_filtered.txt"),
    "cub":       dict(dataset="cub", backbone="resnet18_cub", concept_set="data/concept_sets/cub_filtered.txt",
                       feature_layer="features.final_pool", clip_cutoff=0.26, n_iters=5000, lam=0.0002),
    "places365": dict(dataset="places365", backbone="resnet50", concept_set="data/concept_sets/places365_filtered.txt",
                       clip_cutoff=0.28, n_iters=80, lam=0.0003),
    "imagenet":  dict(dataset="imagenet", backbone="resnet50", concept_set="data/concept_sets/imagenet_filtered.txt",
                       clip_cutoff=0.28, n_iters=80, lam=0.0001),
}

STANDARD_SPARSE_CONFIGS = {
    "cifar10":   dict(dataset="cifar10"),
    "cifar100":  dict(dataset="cifar100", lam=0.003),
    "cub":       dict(dataset="cub", backbone="resnet18_cub", feature_layer="features.final_pool",
                       lam=0.00002, n_iters=5000),
    "places365": dict(dataset="places365", backbone="resnet50", lam=0.0007, n_iters=80),
    "imagenet":  dict(dataset="imagenet", backbone="resnet50", lam=0.0001, n_iters=80),
}

# Published numbers (authors' README). The paper's plain "Standard" row is a
# black-box reference ceiling; the authors provide no training command for it.
PAPER_RESULTS = {
    "lfcbm":            {"cifar10": 86.37, "cifar100": 65.27, "cub": 74.59, "places365": 43.71, "imagenet": 71.98},
    "standard":         {"cifar10": 88.80, "cifar100": 70.10, "cub": 76.70, "places365": 48.56, "imagenet": 76.13},
    "standard_sparse":  {"cifar10": 82.96, "cifar100": 58.34, "cub": 75.96, "places365": 38.46, "imagenet": 74.35},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_default_args(overrides: dict) -> SimpleNamespace:
    """Mirrors the argparse defaults in original_lfcbm/train_cbm.py / train_standard.py."""
    defaults = dict(
        dataset="cifar10", concept_set=None, backbone="clip_RN50", clip_name="ViT-B/16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=512, saga_batch_size=256, proj_batch_size=50000,
        feature_layer="layer4", activation_dir="saved_activations", save_dir="saved_models",
        clip_cutoff=0.25, proj_steps=1000, interpretability_cutoff=0.45, lam=0.0007,
        n_iters=1000, **{"print": False},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LFCBM_CONFIGS))
    p.add_argument("--model", default="lfcbm", choices=["lfcbm", "standard_sparse"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=None,
                    help="activation-caching batch size (paper default 512 is too big for 6-8GB GPUs; try 64, or 32 for Places365/ImageNet)")
    p.add_argument("--saga_batch_size", type=int, default=None)
    p.add_argument("--imagenet_root", default="D:/datasets/ImageNet",
                    help="folder containing train/ and val/ in ImageFolder layout (only used for --dataset imagenet)")
    p.add_argument("--no_compat", action="store_true",
                    help="run the authors' code exactly as published (Windows crashes / RAM limits will return)")
    args = p.parse_args()

    os.chdir(os.path.join(ROOT, "original_lfcbm"))
    sys.path.insert(0, os.getcwd())
    compat.apply(enabled=not args.no_compat, dataset=args.dataset, imagenet_root=args.imagenet_root)

    set_seed(args.seed)
    table = LFCBM_CONFIGS if args.model == "lfcbm" else STANDARD_SPARSE_CONFIGS
    over = dict(table[args.dataset])
    if args.batch_size is not None:
        over["batch_size"] = args.batch_size
    if args.saga_batch_size is not None:
        over["saga_batch_size"] = args.saga_batch_size
    run_args = build_default_args(over)

    print(f"=== Running {args.model} on {args.dataset} (seed={args.seed}) ===")
    print("Config:", vars(run_args))
    start = time.time()
    if args.model == "lfcbm":
        import train_cbm
        train_cbm.train_cbm_and_save(run_args)
    else:
        import train_standard
        train_standard.train_standard(run_args)
    elapsed = time.time() - start

    log = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed, "config": vars(run_args),
        "compat_enabled": not args.no_compat, "elapsed_seconds": elapsed,
        "paper_reported_accuracy": PAPER_RESULTS["lfcbm" if args.model == "lfcbm" else "standard_sparse"][args.dataset],
        "note": "Fill in 'reproduced_accuracy' (see metrics.txt in the saved model folder, key acc_val, or evaluate_cbm.ipynb).",
        "reproduced_accuracy": None,
    }
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}_{args.dataset}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Run config + timing logged to {out_path}")


if __name__ == "__main__":
    main()
