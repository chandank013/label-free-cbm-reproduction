"""
Phase 2 -- Concept Quality Filtering for Label-Free CBM.

Pure numpy (no torch needed), so it can be unit-tested anywhere and run on
CPU in seconds on the CLIP concept-score matrix that LF-CBM already caches.

Notation
    X : (N, K) CLIP image-text similarity scores (N images, K concepts) --
        the `clip_features` matrix inside the authors' train_cbm.py
    y : (N,)   integer class labels of the TRAINING images
    Z : calibrated scores. If center_rows=True, Z = X - rowmean(X): CLIP
        cosine scores share a large image-level offset, which would make
        every pair of concepts look correlated. Removing it first makes
        correlations reflect concept-specific signal.

Two concept-quality signals, both computed on TRAIN data only (so the
filter never sees the validation/test split):

  relevance(k)  = eta^2 = SS_between / SS_total of concept k's score across
                  classes (ANOVA effect size / correlation ratio).
                  ~0  -> the concept scores the same for every class
                         (e.g. "animal", "nature", "beautiful") -> irrelevant
                  ->1 -> the concept almost determines the class.

  redundancy(j,k) = |Pearson corr| of concepts j and k across images, i.e.
                  two concepts that light up on the same images are
                  interchangeable ("long beak" / "large beak").
                  (This is measured in IMAGE-ACTIVATION space, not text
                  space -- which is what the authors' own text-similarity
                  filter cannot see.)

Selection (`greedy_select`) is a greedy relevance-minus-redundancy rule:
eta^2 is the relevance term and the MAX |corr| to an already-chosen concept is
the redundancy term. It is a close relative of minimum-redundancy /
maximum-relevance feature selection (Peng, Long & Ding, IEEE TPAMI 2005,
which averages the redundancy term) and of discriminability + coverage
selection for concept bottlenecks in LaBo (Yang et al., CVPR 2023). It yields
a single concept ordering, so any budget K is a prefix of the same ranking
(nested sets -> clean accuracy-vs-#concepts curves).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# sufficient statistics (chunked so large N never needs N x K x C memory)
# --------------------------------------------------------------------------
def calibrate(X: np.ndarray, center_rows: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=1, keepdims=True) if center_rows else X


def suff_stats(X, y, n_classes, center_rows=True, chunk=8192):
    """One pass over the data. Returns N, per-class counts (C,), per-class
    sums (C,K), total sum (K,), total sum of squares (K,), Gram matrix (K,K)."""
    X = np.asarray(X)
    y = np.asarray(y)
    N, K = X.shape
    counts = np.zeros(n_classes)
    S = np.zeros((n_classes, K))
    T = np.zeros(K)
    Q = np.zeros(K)
    G = np.zeros((K, K))
    for a in range(0, N, chunk):
        Z = calibrate(X[a:a + chunk], center_rows)
        yy = y[a:a + chunk]
        onehot = np.zeros((len(yy), n_classes))
        onehot[np.arange(len(yy)), yy] = 1.0
        counts += onehot.sum(0)
        S += onehot.T @ Z
        T += Z.sum(0)
        Q += (Z * Z).sum(0)
        G += Z.T @ Z
    return N, counts, S, T, Q, G


def relevance_from_stats(N, counts, S, T, Q):
    """eta^2 per concept, in [0, 1]."""
    mean = T / N
    ss_tot = Q - N * mean ** 2
    nz = counts > 0
    ss_between = (S[nz] ** 2 / counts[nz, None]).sum(0) - N * mean ** 2
    eta2 = ss_between / np.maximum(ss_tot, EPS)
    return np.clip(np.nan_to_num(eta2), 0.0, 1.0)


def abs_corr_from_stats(N, T, Q, G):
    """|Pearson correlation| matrix (K,K), zero diagonal."""
    mean = T / N
    cov = G / N - np.outer(mean, mean)
    sd = np.sqrt(np.maximum(np.diag(cov), EPS))
    corr = np.abs(cov / np.outer(sd, sd))
    np.fill_diagonal(corr, 0.0)
    return np.clip(np.nan_to_num(corr), 0.0, 1.0)


def score_concepts(X, y, n_classes, center_rows=True):
    """Convenience: (relevance (K,), abs_corr (K,K)) from a score matrix."""
    N, counts, S, T, Q, G = suff_stats(X, y, n_classes, center_rows)
    return relevance_from_stats(N, counts, S, T, Q), abs_corr_from_stats(N, T, Q, G)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def greedy_select(relevance, abs_corr, k, w_rel=1.0, w_red=1.0, agg="max"):
    """Greedy relevance-minus-redundancy ordering; returns the first k indices.

        score(c) = w_rel * rel(c)/max(rel)  -  w_red * R(c, S)

    where S is the set already selected and R is
        agg="max"  : max_{s in S} |corr(c,s)|   (default; penalises the single
                     nearest duplicate -- in the spirit of Maximal Marginal Relevance)
        agg="mean" : mean_{s in S} |corr(c,s)|  (the original mRMR form; the
                     1/|S| factor dilutes a lone duplicate as S grows, so it
                     lets near-copies through -- kept only as an ablation)

    w_red=0          -> relevance-only ranking
    w_rel=0          -> redundancy-only (pure diversity) ranking
    w_rel=w_red=1    -> the proposed filter
    """
    K = len(relevance)
    k = int(min(max(k, 1), K))
    rel = relevance / max(relevance.max(), EPS)
    selected = []
    red_max, red_sum = np.zeros(K), np.zeros(K)
    available = np.ones(K, dtype=bool)
    for _ in range(k):
        red = red_max if agg == "max" else red_sum / max(len(selected), 1)
        score = w_rel * rel - w_red * red
        score[~available] = -np.inf
        j = int(np.argmax(score))
        selected.append(j)
        available[j] = False
        red_max = np.maximum(red_max, abs_corr[:, j])
        red_sum += abs_corr[:, j]
    return np.array(selected)


mrmr_select = greedy_select  # backwards-compatible alias


def clip_topk_select(X, k, topn=5, chunk=8192):
    """Baseline mirroring the authors' own criterion: rank concepts by the mean
    of their top-5 CLIP activations ('does this concept ever show up?').
    Chunked running top-n, so memory stays O(topn x K) even for 1.8M images
    (a full np.sort would copy the whole N x K matrix)."""
    X = np.asarray(X)
    N, K = X.shape
    top = np.full((topn, K), -np.inf)
    for a in range(0, N, chunk):
        cat = np.vstack([top, X[a:a + chunk].astype(np.float64, copy=False)])
        top = np.partition(cat, cat.shape[0] - topn, axis=0)[-topn:]
    valid = np.isfinite(top)
    score = np.where(valid, top, 0.0).sum(0) / np.maximum(valid.sum(0), 1)
    return np.argsort(-score, kind="stable")[:int(k)]


def random_select(K, k, seed):
    return np.random.RandomState(seed).choice(K, size=int(k), replace=False)


def to_mask(idx, K):
    m = np.zeros(K, dtype=bool)
    m[np.asarray(idx, dtype=int)] = True
    return m


def explain_removed(concepts, keep_mask, relevance, abs_corr):
    """For every removed concept: why (low relevance / duplicate of which kept
    concept). Used for qualitative examples in the report."""
    kept = np.where(keep_mask)[0]
    rows = []
    for i in np.where(~keep_mask)[0]:
        j = kept[np.argmax(abs_corr[i, kept])] if len(kept) else -1
        rows.append(dict(concept=concepts[i], relevance=float(relevance[i]),
                         nearest_kept=concepts[j] if j >= 0 else "",
                         corr_with_nearest_kept=float(abs_corr[i, j]) if j >= 0 else 0.0))
    return rows


# --------------------------------------------------------------------------
# evaluation metrics for a FINAL concept set (measured on held-out features)
# --------------------------------------------------------------------------
def nn_similarity(M):
    """Per-concept max similarity to any OTHER concept. M: (K,K), diag ignored."""
    M = np.array(M, dtype=np.float64)
    np.fill_diagonal(M, -np.inf)
    return M.max(1) if M.shape[0] > 1 else np.zeros(M.shape[0])


def quality_report(Xv, yv, n_classes, text_emb=None, center_rows=True, dup_corr=0.8, dup_text=0.9):
    """Metrics for the concepts whose held-out scores are the columns of Xv.

    text_emb : optional (K, D) L2-normalised CLIP text embeddings of those concepts
    Returns (summary dict, per-concept dict of arrays)
    """
    rel, corr = score_concepts(Xv, yv, n_classes, center_rows)
    nn_corr = nn_similarity(corr)
    K = len(rel)
    iu = np.triu_indices(K, 1)
    summary = dict(
        n_concepts=int(K),
        relevance_mean=float(rel.mean()),
        relevance_median=float(np.median(rel)),
        redundancy_mean_abs_corr=float(corr[iu].mean()) if K > 1 else 0.0,
        redundancy_nn_abs_corr=float(nn_corr.mean()),
        frac_concepts_with_dup_act=float((nn_corr > dup_corr).mean()),
    )
    per = dict(relevance_heldout=rel, nn_abs_corr_heldout=nn_corr)
    if text_emb is not None and K > 1:
        sim = np.asarray(text_emb) @ np.asarray(text_emb).T
        nn_txt = nn_similarity(sim)
        summary.update(redundancy_mean_text_cos=float(sim[iu].mean()),
                       redundancy_nn_text_cos=float(nn_txt.mean()),
                       frac_concepts_with_dup_text=float((nn_txt > dup_text).mean()))
        per["nn_text_cos"] = nn_txt
    return summary, per


# --------------------------------------------------------------------------
# glue used by the generated trainer (kept in numpy so it is unit-testable)
# --------------------------------------------------------------------------
FILTER_MODES = ("all", "greedy", "clip_topk", "random")


def run_filter(mode, X_train, y_train, concepts, n_classes, keep_frac=1.0, seed=0,
               w_rel=1.0, w_red=1.0, agg="max", center_rows=True):
    """Decide which concepts survive. Uses TRAIN data only.

    Returns (keep_mask (K,) bool, diagnostics dict). mode='all' keeps everything,
    so the baseline runs through exactly the same code path as the filtered runs.
    """
    K = len(concepts)
    diag = dict(filter_mode=mode, n_before_filter=K, keep_frac=float(keep_frac))
    if mode not in FILTER_MODES:
        raise ValueError(f"filter_mode must be one of {FILTER_MODES}, got {mode!r}")
    if mode == "all":
        diag.update(n_after_filter=K, removed=[])
        return np.ones(K, dtype=bool), diag
    k = max(1, int(round(keep_frac * K)))
    rel, corr = score_concepts(X_train, y_train, n_classes, center_rows)
    if mode == "greedy":
        sel = greedy_select(rel, corr, k, w_rel=w_rel, w_red=w_red, agg=agg)
    elif mode == "clip_topk":
        sel = clip_topk_select(X_train, k)
    else:
        sel = random_select(K, k, seed)
    keep = to_mask(sel, K)
    diag.update(n_after_filter=int(keep.sum()), removed=explain_removed(concepts, keep, rel, corr))
    return keep, diag


def final_report(Xv, yv, n_classes, concepts, W_g, text_emb=None, center_rows=True):
    """Summary + per-concept rows for the FINAL concept set of a trained model.

    Xv : (Nv, K) held-out CLIP scores of the final concepts
    W_g: (n_classes, K) final sparse layer weights (numpy)
    NEC (number of effective concepts, VLG-CBM, NeurIPS 2024) = average number
    of non-zero weights per class = nnz / n_classes.
    """
    summary, per = quality_report(Xv, yv, n_classes, text_emb=text_emb, center_rows=center_rows)
    used = (np.abs(W_g) > 1e-5)
    per["n_classes_using"] = used.sum(0)
    nnz = int(used.sum())
    summary.update(nnz=nnz, NEC=nnz / n_classes, frac_concepts_unused=float((per["n_classes_using"] == 0).mean()))
    rows = []
    for i, c in enumerate(concepts):
        row = dict(concept=c)
        row.update({k: (float(v[i]) if k != "n_classes_using" else int(v[i])) for k, v in per.items()})
        rows.append(row)
    return summary, rows
