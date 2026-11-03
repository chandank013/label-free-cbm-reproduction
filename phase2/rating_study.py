"""
Blinded human rating study -- the INDEPENDENT check on concept quality.

Why: the filter selects on eta^2 (statistical relevance). Reporting eta^2 as
the success metric would be circular. People judging the actual concept
text are a measure the filter never optimised.

Design: raters see a shuffled mix of concepts the filter KEPT and concepts it
REMOVED (disjoint sets, source hidden). If the filter works, kept concepts should
be rated more relevant than removed ones.

Step 1 (one person):  export a shuffled, source-hidden sheet
    python phase2/rating_study.py export --dataset cub --seed 0 --keep 50 --n 60
    -> results/phase2/rating/<dataset>_rating_sheet.csv   (give this to every rater)
    -> results/phase2/rating/<dataset>_rating_key.csv     (KEEP PRIVATE until scoring)

Step 2 (each rater, independently, e.g. the 5 team members):
    fill  relevance_1to5   1 = useless for telling the classes apart ("animal", "nature")
                           5 = clearly useful, visual, specific
    optionally fill  duplicate_of_id  id of another concept on THIS sheet that says the same thing
    save as  <dataset>_rated_<yourname>.csv  in the same folder

Step 3:
    python phase2/rating_study.py score --dataset cub
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RDIR = os.path.join(ROOT, "results", "phase2", "rating")


def export(a):
    tdir = os.path.join(ROOT, "results", "phase2", a.dataset, "tables")
    base = pd.read_csv(os.path.join(tdir, f"{a.dataset}_baseline_k100_s{a.seed}.csv"))["concept"].drop_duplicates()
    kept = set(pd.read_csv(os.path.join(tdir, f"{a.dataset}_greedy_k{a.keep}_s{a.seed}.csv"))["concept"])
    pools = {"kept": base[base.isin(kept)], "removed": base[~base.isin(kept)]}
    if min(len(v) for v in pools.values()) == 0:
        raise SystemExit("kept or removed pool is empty -- check the budget/seed you passed")
    rs = np.random.RandomState(12345)
    frames = [v.sample(n=min(a.n, len(v)), random_state=rs).to_frame("concept").assign(source=k) for k, v in pools.items()]
    allc = pd.concat(frames).sample(frac=1.0, random_state=rs).reset_index(drop=True)
    allc.insert(0, "id", np.arange(1, len(allc) + 1))
    os.makedirs(RDIR, exist_ok=True)
    keyp = os.path.join(RDIR, f"{a.dataset}_rating_key.csv")
    sheetp = os.path.join(RDIR, f"{a.dataset}_rating_sheet.csv")
    allc.to_csv(keyp, index=False)
    allc[["id", "concept"]].assign(relevance_1to5="", duplicate_of_id="").to_csv(sheetp, index=False)
    print(f"{len(allc)} concepts ({', '.join(f'{k}: {min(a.n, len(v))}' for k, v in pools.items())}) -> {sheetp}\nkey (keep private) -> {keyp}")


def score(a):
    from scipy.stats import mannwhitneyu
    key = pd.read_csv(os.path.join(RDIR, f"{a.dataset}_rating_key.csv"))
    files = glob.glob(os.path.join(RDIR, f"{a.dataset}_rated_*.csv"))
    if not files:
        raise SystemExit("no rated sheets found (expected <dataset>_rated_<name>.csv)")
    rows = []
    for f in files:
        r = pd.read_csv(f).merge(key[["id", "source"]], on="id")
        r["rater"] = os.path.basename(f)
        rows.append(r)
    d = pd.concat(rows)
    d["relevance_1to5"] = pd.to_numeric(d["relevance_1to5"], errors="coerce")
    d["is_dup"] = d["duplicate_of_id"].notna() & (d["duplicate_of_id"].astype(str).str.strip() != "")
    per = d.groupby(["source", "id"]).agg(rel=("relevance_1to5", "mean"), dup=("is_dup", "mean")).reset_index()
    print(f"{len(files)} raters, {per.id.nunique()} concepts\n")
    for src, g in per.groupby("source"):
        print(f"{src:9s} n={len(g):3d}  mean relevance {g.rel.mean():.2f} ± {g.rel.std():.2f}   "
              f"share rated 'useless' (<=2): {(g.rel <= 2).mean():.0%}   flagged duplicate: {g.dup.mean():.0%}")
    k_, r_ = per[per.source == "kept"].rel.dropna(), per[per.source == "removed"].rel.dropna()
    if len(k_) and len(r_):
        u = mannwhitneyu(k_, r_, alternative="greater")
        print(f"\nMann-Whitney U (kept > removed relevance): U={u.statistic:.0f}, p={u.pvalue:.4f}")
    if len(files) > 1:                                   # crude agreement: mean pairwise Spearman
        piv = d.pivot_table(index="id", columns="rater", values="relevance_1to5")
        cors = piv.corr(method="spearman").values[np.triu_indices(piv.shape[1], 1)]
        print(f"inter-rater agreement (mean pairwise Spearman): {np.nanmean(cors):.2f}")
    print("\nNote: the duplicate flag only sees concepts present in this sheet, so treat it as weak evidence; "
          "the quantitative redundancy metrics are the primary redundancy result.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--dataset", required=True); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--keep", type=int, default=50); e.add_argument("--n", type=int, default=60); e.set_defaults(fn=export)
    s = sp.add_parser("score"); s.add_argument("--dataset", required=True); s.set_defaults(fn=score)
    a = ap.parse_args(); a.fn(a)
