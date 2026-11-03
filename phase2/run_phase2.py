"""
Phase 2 experiment runner -- all five datasets.

For one dataset it trains, for every seed:
    baseline   : the original LF-CBM (filter_mode=all)                  1 run
    greedy     : OUR relevance + redundancy filter, per budget
    random     : same NUMBER of concepts, chosen at random (control)
    clip_topk  : same number, ranked by the authors' own "does it ever fire"
                 criterion (a stronger, fairer baseline)

Budgets are FRACTIONS of the concepts that survive the authors' clip_cutoff and
are fixed in advance -- never tune them on the validation split (for CIFAR/CUB
the authors' "val" split IS the test split).

Presets (override with --seeds / --budgets):
    cifar10, cifar100, cub : seeds 0 1 2, budgets 0.75 0.5 0.25   -> 30 runs each
    places365, imagenet    : seed 0,      budget  0.5             ->  4 runs each
The two large datasets get a reduced protocol because every run repeats
projection learning + the sparse solver over ~1.3-1.8 M images; with one seed
there are NO error bars, and the report must say so.

    python phase2/run_phase2.py --dataset cub --batch_size 64
    python phase2/run_phase2.py --dataset cifar10 --batch_size 64
    python phase2/run_phase2.py --dataset places365 --batch_size 32
    python phase2/run_phase2.py --dataset imagenet --batch_size 32 --imagenet_root D:/datasets/ImageNet
    python phase2/run_phase2.py --dataset cub --dry_run          # print the plan only

Resumable: a run whose result JSON exists is skipped (--force to redo). Cached
CLIP / backbone activations are reused across runs, so only the first run per
dataset pays the feature-extraction cost. Ablations: --ablate.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.path.join(ROOT, "results", "phase2")

PRESETS = {
    "small": dict(seeds=[0, 1, 2], budgets=[0.75, 0.5, 0.25]),
    "large": dict(seeds=[0], budgets=[0.5]),
}
SIZE = {"cifar10": "small", "cifar100": "small", "cub": "small", "places365": "large", "imagenet": "large"}
ALL_METHODS = ["baseline", "greedy", "random", "clip_topk"]


def plan(dataset, seeds, budgets, methods, ablate):
    runs = []
    for seed in seeds:
        base = dict(seed=seed, w_rel=1.0, w_red=1.0, agg="max")
        if "baseline" in methods:
            runs.append(dict(base, method="baseline", mode="all", frac=1.0))
        for frac in budgets:
            for m in ("greedy", "random", "clip_topk"):
                if m in methods:
                    runs.append(dict(base, method=m, mode=m, frac=frac))
            if ablate:
                runs.append(dict(base, method="abl_relevance_only", mode="greedy", frac=frac, w_red=0.0))
                runs.append(dict(base, method="abl_diversity_only", mode="greedy", frac=frac, w_rel=0.0))
                runs.append(dict(base, method="abl_mean_agg", mode="greedy", frac=frac, agg="mean"))
    for r in runs:
        r["tag"] = f"_{r['method']}_k{int(round(r['frac'] * 100))}_s{r['seed']}"
        r["name"] = f"{dataset}{r['tag']}"
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SIZE))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--budgets", type=float, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--saga_batch_size", type=int, default=None)
    ap.add_argument("--imagenet_root", default="D:/datasets/ImageNet")
    ap.add_argument("--no_compat", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    preset = PRESETS[SIZE[args.dataset]]
    seeds = args.seeds if args.seeds is not None else preset["seeds"]
    budgets = args.budgets if args.budgets is not None else preset["budgets"]
    runs = plan(args.dataset, seeds, budgets, args.methods, args.ablate)
    out_dir = os.path.join(OUT_ROOT, args.dataset)
    todo = [r for r in runs if args.force or not os.path.exists(os.path.join(out_dir, r["name"] + ".json"))]
    print(f"{args.dataset}: preset '{SIZE[args.dataset]}' seeds={seeds} budgets={budgets}")
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to run")
    if len(seeds) == 1:
        print("NOTE: a single seed gives no error bars; say so in the report.")
    if args.dry_run:
        for r in todo:
            print("  ", r["name"])
        return
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    sys.path.insert(0, ROOT)
    import compat                                                       # noqa: E402
    os.chdir(os.path.join(ROOT, "original_lfcbm"))
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(ROOT, "phase2"))
    compat.apply(enabled=not args.no_compat, dataset=args.dataset, imagenet_root=args.imagenet_root)
    from run_experiment import LFCBM_CONFIGS, build_default_args, set_seed   # noqa: E402
    import train_cbm_filtered                                                  # noqa: E402

    for i, r in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {r['name']} ===", flush=True)
        set_seed(r["seed"])
        over = dict(LFCBM_CONFIGS[args.dataset], batch_size=args.batch_size)
        if args.saga_batch_size:
            over["saga_batch_size"] = args.saga_batch_size
        a = build_default_args(over)
        a.filter_mode, a.keep_frac, a.seed, a.tag = r["mode"], r["frac"], r["seed"], r["tag"]
        a.w_rel, a.w_red, a.agg, a.center_rows = r["w_rel"], r["w_red"], r["agg"], True
        t0 = time.time()
        train_cbm_filtered.train_cbm_and_save(a)
        dirs = sorted(glob.glob(f"saved_models/{args.dataset}_cbm_*{r['tag']}"), key=os.path.getmtime)
        if not dirs:
            raise RuntimeError(f"no saved model folder found for {r['name']}")
        rep = json.load(open(os.path.join(dirs[-1], "phase2_report.json")))
        rep["summary"].update(method=r["method"], tag=r["tag"], runtime_s=time.time() - t0, model_dir=dirs[-1])
        json.dump(rep, open(os.path.join(out_dir, r["name"] + ".json"), "w"), indent=2)
        shutil.copy(os.path.join(dirs[-1], "concept_table.csv"), os.path.join(out_dir, "tables", r["name"] + ".csv"))
        print(f"saved -> {os.path.join(out_dir, r['name'] + '.json')}", flush=True)


if __name__ == "__main__":
    main()
