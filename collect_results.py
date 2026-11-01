"""
Aggregates results/*.json (written by run_experiment.py, then filled in with
'reproduced_accuracy' after you run evaluate_cbm.ipynb) into the final
paper-vs-reproduced comparison table for your Phase 1 report.

Usage:
    python collect_results.py
    python collect_results.py --out results/comparison_table.md
"""
import argparse
import glob
import json
import os

DATASETS = ["cifar10", "cifar100", "cub", "places365", "imagenet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/comparison_table.md")
    args = ap.parse_args()

    rows = {}
    for path in glob.glob(os.path.join(args.results_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        key = (r["model"], r["dataset"])
        rows[key] = r

    lines = [
        "| Model | Dataset | Paper Reported (%) | Reproduced (%) | Difference | Seed | Runtime (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in ["lfcbm", "standard_sparse"]:
        for ds in DATASETS:
            r = rows.get((model, ds))
            if r is None:
                lines.append(f"| {model} | {ds} | — | *not run* | — | — | — |")
                continue
            paper = r.get("paper_reported_accuracy")
            repro = r.get("reproduced_accuracy")
            diff = f"{repro - paper:+.2f}" if (paper is not None and repro is not None) else "—"
            repro_str = f"{repro:.2f}" if repro is not None else "*fill in*"
            paper_str = f"{paper:.2f}" if paper is not None else "—"
            lines.append(
                f"| {model} | {ds} | {paper_str} | {repro_str} | {diff} | "
                f"{r['seed']} | {r['elapsed_seconds']:.0f} |"
            )

    table = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# LF-CBM Replication: Paper vs. Reproduced Results\n\n" + table + "\n")
    print(table)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
