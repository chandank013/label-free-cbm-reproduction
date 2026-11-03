"""
Aggregate results/phase2/<dataset>/*.json into:
  results/phase2/<dataset>_summary.md        mean +- std tables (Original vs Improved + controls)
  results/phase2/<dataset>_curves.png        accuracy / redundancy / relevance vs #concepts

    python phase2/aggregate_phase2.py --dataset cub
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = [("acc_val", "Accuracy (%)", 100.0), ("n_concepts", "# concepts", 1.0), ("NEC", "NEC", 1.0),
           ("redundancy_nn_abs_corr", "Redundancy (act.)", 1.0), ("redundancy_nn_text_cos", "Redundancy (text)", 1.0),
           ("relevance_mean", "Relevance", 1.0), ("frac_concepts_unused", "Unused concepts", 1.0)]


def load(results_dir, dataset):
    rows = []
    for p in glob.glob(os.path.join(results_dir, dataset, "*.json")):
        s = json.load(open(p))["summary"]
        rows.append(s)
    if not rows:
        raise SystemExit(f"no result JSONs under {os.path.join(results_dir, dataset)}")
    df = pd.DataFrame(rows)
    df["frac"] = df["keep_frac"].round(2)
    return df


def ms(g, col, scale):
    if col not in g or g[col].isna().all():
        return "n/a"
    v = g[col].astype(float) * scale
    if len(v) < 2:                                    # one seed: no error bar to report
        return f"{v.mean():.2f}" if col != "n_concepts" else f"{v.mean():.0f}"
    sd = v.std(ddof=1)
    return f"{v.mean():.2f} ± {sd:.2f}" if col != "n_concepts" else f"{v.mean():.0f} ± {sd:.0f}"


def table(df, methods):
    head = "| Method | Budget | n | " + " | ".join(m[1] for m in METRICS) + " |\n"
    head += "|---|---|---|" + "---|" * len(METRICS) + "\n"
    out = head
    for method in methods:
        for frac, g in df[df.method == method].groupby("frac"):
            out += f"| {method} | {frac:.2f} | {len(g)} | " + " | ".join(ms(g, c, sc) for c, _, sc in METRICS) + " |\n"
    return out


def paired(df, a, b, col="acc_val"):
    """Seed-paired difference a - b at each budget (in accuracy points)."""
    lines = ["| Budget | " + f"{a} − {b} (acc. points)" + " | seeds | wins |", "|---|---|---|---|"]
    for frac in sorted(df[df.method == a].frac.unique()):
        x = df[(df.method == a) & (df.frac == frac)].set_index("seed")[col]
        y = df[(df.method == b) & (df.frac == frac)].set_index("seed")[col]
        d = (x - y).dropna() * 100
        if len(d):
            pm = f"{d.mean():+.2f} ± {d.std(ddof=1):.2f}" if len(d) > 1 else f"{d.mean():+.2f} (1 seed)"
            lines.append(f"| {frac:.2f} | {pm} | {len(d)} | {(d > 0).sum()}/{len(d)} |")
    return "\n".join(lines)


def figure(df, path, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    style = {"greedy": ("tab:red", "o", "Ours (relevance + redundancy)"), "clip_topk": ("tab:blue", "s", "Top-k by CLIP activation"),
             "random": ("tab:gray", "^", "Random (control)")}
    base = df[df.method == "baseline"]
    panels = [("acc_val", "Accuracy", 100.0), ("redundancy_nn_abs_corr", "Redundancy: mean nearest-neighbour |corr| (lower = better)", 1.0),
              ("relevance_mean", "Relevance: mean η² (higher = better)", 1.0)]
    for a, (col, title, sc) in zip(ax, panels):
        for m, (c, mk, lab) in style.items():
            g = df[df.method == m].groupby("frac").agg(x=("n_concepts", "mean"), y=(col, "mean"), e=(col, "std"))
            if len(g):
                a.errorbar(g.x, g.y * sc, yerr=(g.e.fillna(0) * sc), color=c, marker=mk, capsize=3, label=lab)
        if len(base):
            a.errorbar([base.n_concepts.mean()], [base[col].mean() * sc], yerr=[base[col].std(ddof=0) * sc], color="black",
                       marker="*", ms=12, capsize=3, label="Original LF-CBM", linestyle="none")
        a.set_xlabel("# concepts in final model"); a.set_title(title, fontsize=9); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: concept filtering vs matched-budget controls (mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


DATASETS = ["cifar10", "cifar100", "cub", "places365", "imagenet"]


def headline(results_dir, budget=0.5):
    """One table: original vs ours vs controls at one budget, across every dataset that has results."""
    cols = [("acc_val", "Acc (%)", 100.0), ("n_concepts", "# concepts", 1.0), ("redundancy_nn_abs_corr", "Redund. (act.)", 1.0),
            ("redundancy_nn_text_cos", "Redund. (text)", 1.0), ("relevance_mean", "Relevance η²", 1.0), ("NEC", "NEC", 1.0)]
    out = f"# Phase 2 headline: original LF-CBM vs improved, budget {budget:.2f}, all datasets\n\n"
    out += "| Dataset | Method | seeds | " + " | ".join(c[1] for c in cols) + " |\n|---|---|---|" + "---|" * len(cols) + "\n"
    for ds in DATASETS:
        if not glob.glob(os.path.join(results_dir, ds, "*.json")):
            out += f"| {ds} | *no results yet* | | " + " | ".join("" for _ in cols) + " |\n"
            continue
        df = load(results_dir, ds)
        for method, frac in [("baseline", 1.0), ("greedy", budget), ("random", budget), ("clip_topk", budget)]:
            g = df[(df.method == method) & ((df.frac - frac).abs() < 1e-6)]
            if len(g):
                out += f"| {ds} | {method} | {g.seed.nunique()} | " + " | ".join(ms(g, c, sc) for c, _, sc in cols) + " |\n"
    out += "\nSingle-seed rows have no ± (no error bars). Accuracy is on the authors' `val` split (= test split for CIFAR/CUB).\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--all", action="store_true", help="headline table across all five datasets")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "results", "phase2"))
    args = ap.parse_args()
    if args.all:
        md = headline(args.results_dir, args.budget)
        out = os.path.join(args.results_dir, "ALL_datasets_summary.md")
        open(out, "w").write(md); print(md); print("wrote", out)
        return
    if not args.dataset:
        ap.error("give --dataset NAME or --all")
    df = load(args.results_dir, args.dataset)
    abl = [m for m in df.method.unique() if m.startswith("abl_")]
    md = f"# Phase 2 results — {args.dataset}\n\n"
    md += "## Original LF-CBM vs improved (ours) at pre-registered budgets\n\n" + table(df, ["baseline", "greedy"]) + "\n"
    md += "## Matched-budget controls\n\n" + table(df, ["random", "clip_topk"]) + "\n"
    md += "## Paired differences (same seed, same budget)\n\n"
    md += "**Ours − random**\n\n" + paired(df, "greedy", "random") + "\n\n"
    md += "**Ours − top-k by CLIP activation**\n\n" + paired(df, "greedy", "clip_topk") + "\n\n"
    if abl:
        md += "## Ablations\n\n" + table(df, abl) + "\n"
    md += ("\n*Redundancy (act.)* / *(text)* = mean over concepts of the highest similarity to any other kept concept, in image-activation / CLIP-text space. "
           "*Relevance* = mean η² on held-out features. *NEC* = non-zero final-layer weights per class. "
           "Accuracy is on the authors' `val` split (which is the test split for CIFAR/CUB).\n")
    out = os.path.join(args.results_dir, f"{args.dataset}_summary.md")
    open(out, "w").write(md)
    figure(df, os.path.join(args.results_dir, f"{args.dataset}_curves.png"), args.dataset)
    print(md); print("wrote", out)


if __name__ == "__main__":
    main()
