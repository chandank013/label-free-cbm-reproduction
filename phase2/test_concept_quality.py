"""
Unit tests for concept_quality.py on SYNTHETIC data where the ground truth is
known: some concepts are planted as class-irrelevant, some as near-duplicates.
Run:  python phase2/test_concept_quality.py      (or: pytest phase2/)

These tests prove the algorithm does what it claims on data we control. They
say NOTHING about real CLIP scores -- that is what the real experiments are for.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import concept_quality as cq

C, K_INF, K_IRR, K_DUP = 10, 30, 10, 20
K = K_INF + K_IRR + K_DUP
INF = np.arange(0, K_INF)
IRR = np.arange(K_INF, K_INF + K_IRR)
DUP = np.arange(K_INF + K_IRR, K)          # DUP[i] duplicates INF[i], i < K_DUP
MU = np.random.RandomState(123).normal(0, 1.0, size=(C, K_INF))


def make_data(n, seed, shared_offset=3.0):
    rs = np.random.RandomState(seed)
    y = rs.randint(0, C, size=n)
    X = np.zeros((n, K))
    X[:, INF] = MU[y] + rs.normal(0, 1.0, size=(n, K_INF))            # informative
    X[:, IRR] = 0.3 + rs.normal(0, 1.0, size=(n, K_IRR))              # class-agnostic
    X[:, DUP] = X[:, INF[:K_DUP]] + rs.normal(0, 0.2, size=(n, K_DUP))  # near-copies
    X += rs.normal(0, shared_offset, size=(n, 1))                     # CLIP-style shared offset
    return X, y


def dup_pairs_kept(sel):
    s = set(sel.tolist())
    return sum(1 for i in range(K_DUP) if INF[i] in s and DUP[i] in s)


def test_chunked_stats_match_direct():
    X, y = make_data(2500, 0)
    rel_a, corr_a = cq.score_concepts(X, y, C)
    N, counts, S, T, Q, G = cq.suff_stats(X, y, C, chunk=333)       # awkward chunk size
    rel_b = cq.relevance_from_stats(N, counts, S, T, Q)
    corr_b = cq.abs_corr_from_stats(N, T, Q, G)
    Z = cq.calibrate(X)
    direct = np.array([sum((y == c).sum() * (Z[y == c, k].mean() - Z[:, k].mean()) ** 2 for c in range(C))
                       / ((Z[:, k] - Z[:, k].mean()) ** 2).sum() for k in range(K)])
    assert np.allclose(rel_a, rel_b) and np.allclose(corr_a, corr_b)
    assert np.allclose(rel_a, direct, atol=1e-9), "eta^2 must match a direct computation"


def test_irrelevant_concepts_get_lowest_relevance():
    X, y = make_data(4000, 1)
    rel, _ = cq.score_concepts(X, y, C)
    bottom = set(np.argsort(rel)[:K_IRR + 2].tolist())
    assert len(bottom & set(IRR.tolist())) >= K_IRR - 1, "planted irrelevant concepts should rank last"
    assert rel[INF].mean() > 5 * rel[IRR].mean()


def test_row_centering_removes_shared_offset():
    X, y = make_data(4000, 2)
    _, corr_raw = cq.score_concepts(X, y, C, center_rows=False)
    _, corr_cen = cq.score_concepts(X, y, C, center_rows=True)
    sub = np.ix_(IRR, IRR)
    off = ~np.eye(K_IRR, dtype=bool)
    assert corr_raw[sub][off].mean() > 0.6, "without centering, independent concepts look correlated"
    assert corr_cen[sub][off].mean() < 0.2, "centering should remove the spurious correlation"
    # genuine duplicates must stay highly correlated after centering
    assert np.mean([corr_cen[INF[i], DUP[i]] for i in range(K_DUP)]) > 0.8


def test_filter_beats_baselines_on_heldout_data():
    Xtr, ytr = make_data(4000, 3)
    Xte, yte = make_data(4000, 4)                      # same concepts, fresh images
    rel, corr = cq.score_concepts(Xtr, ytr, C)
    budget = 30
    ours = cq.greedy_select(rel, corr, budget)
    mean_agg = cq.greedy_select(rel, corr, budget, agg="mean")
    rel_only = cq.greedy_select(rel, corr, budget, w_red=0.0)
    clip = cq.clip_topk_select(Xtr, budget)
    rnd = cq.random_select(K, budget, seed=0)

    rep = {n: cq.quality_report(Xte[:, s], yte, C)[0] for n, s in
           dict(ours=ours, rel_only=rel_only, clip=clip, rnd=rnd).items()}
    assert len(np.intersect1d(ours, IRR)) <= 2, "filter should drop (almost) all irrelevant concepts"
    assert dup_pairs_kept(ours) <= 3
    assert dup_pairs_kept(rel_only) > dup_pairs_kept(ours), "redundancy term must matter"
    assert dup_pairs_kept(mean_agg) > dup_pairs_kept(ours), "mean-aggregation dilutes duplicates (why we use max)"
    for name in ("clip", "rnd"):
        assert rep["ours"]["relevance_mean"] > rep[name]["relevance_mean"]
    assert rep["ours"]["redundancy_nn_abs_corr"] < rep["rnd"]["redundancy_nn_abs_corr"]
    assert rep["ours"]["redundancy_nn_abs_corr"] < rep["rel_only"]["redundancy_nn_abs_corr"]


def test_nested_budgets_and_explanations():
    X, y = make_data(3000, 5)
    rel, corr = cq.score_concepts(X, y, C)
    big, small = cq.greedy_select(rel, corr, 40), cq.greedy_select(rel, corr, 15)
    assert np.array_equal(big[:15], small), "budgets must be prefixes of one ranking"
    keep = cq.to_mask(small, K)
    names = [f"c{i}" for i in range(K)]
    rows = cq.explain_removed(names, keep, rel, corr)
    assert len(rows) == K - 15
    # a removed duplicate whose original was kept should name that original
    for r in rows:
        i = int(r["concept"][1:])
        if i in DUP and INF[i - DUP[0]] in set(small.tolist()):
            assert r["nearest_kept"] == f"c{INF[i - DUP[0]]}"
            assert r["corr_with_nearest_kept"] > 0.8
            break
    else:
        raise AssertionError("expected at least one explained duplicate")


def test_clip_topk_chunked_matches_full_sort():
    X, _ = make_data(1000, 7)
    ref = np.argsort(-np.sort(X, axis=0)[-5:].mean(0), kind="stable")[:25]
    assert np.array_equal(cq.clip_topk_select(X, 25, chunk=37), ref)
    assert np.array_equal(cq.clip_topk_select(X[:3], 5), np.argsort(-X[:3].mean(0), kind="stable")[:5])  # N < topn


def test_text_space_metrics():
    rs = np.random.RandomState(0)
    base = rs.normal(size=(5, 16))
    emb = np.vstack([base, base + 0.01 * rs.normal(size=(5, 16))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    X, y = make_data(1000, 6)
    s, per = cq.quality_report(X[:, :10], y, C, text_emb=emb)
    assert s["frac_concepts_with_dup_text"] == 1.0 and "nn_text_cos" in per


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
