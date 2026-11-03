# Phase 2 — Concept Quality Filtering for Label-Free CBM

**Deadline: Nov 3, 2026.** Deliverables: code updates, before/after experiments, 1-page research-gap explanation.

## 1. What the original already does (so the gap is stated accurately)

LF-CBM already filters concepts. A fair gap statement has to start here:

| Stage | Where | What it checks | Looks at class labels? | Looks at images? |
|---|---|---|---|---|
| Offline text filters (length, similarity to class names, concept–concept text similarity) | before the repo; only the filtered sets ship in `data/concept_sets/` | wording | no | no |
| `clip_cutoff` | `train_cbm.py` | the concept's top-5 CLIP activation is high enough, i.e. it *appears* in the data | no | yes |
| `interpretability_cutoff` | `train_cbm.py` | the learned projection neuron matches the CLIP concept scores | no | yes |

> The repo ships only the *already filtered* concept sets, so the exact text-filter rules are not in the code. Check §3.1 / appendix of the paper before quoting a threshold.

**The gap:** none of these asks (a) whether a concept helps *separate the classes*, or (b) whether two concepts fire on the *same images*. Redundancy is only judged on wording, before any image is seen.

**Evidence the gap exists (proxy, from the shipped sets):** about 26% of CUB concepts (95 of 370) sit in a pair with ≥0.6 word overlap, versus about 2% on CIFAR-10. This is only a proxy: "a black and white head" and "a black head with a white beak" overlap in words but are different concepts. The real test is activation correlation, which Phase 2's first experiment measures on the original concept set.

## 2. Proposed extension

Insert a **Concept Quality Filter** between the `clip_cutoff` step and projection-layer training, using **train data only**:

```
CLIP scores X (N×K) ──► relevance η²(k)      ──┐
                  └──► |corr(j,k)| matrix     ──┴─► greedy selection ──► kept concepts ──► (original pipeline)
```

- **Relevance** of concept *k* = η² (ANOVA effect size): the fraction of its score variance explained by class. ≈0 for "animal"/"nature", high for class-specific cues.
- **Redundancy** of *j,k* = |Pearson correlation| of their scores across images. CLIP scores share an image-level offset, so scores are row-centred first (otherwise unrelated concepts look correlated; this is covered by a unit test).
- **Selection** (greedy, one ranking, any budget is a prefix of it):
  `score(c) = relevance(c)/max − max_{s ∈ selected} |corr(c,s)|`
  The *max* penalty targets the nearest duplicate. The mean-penalty form (original mRMR) lets a lone duplicate through as the selected set grows. The unit tests confirm this, and it is kept as an ablation.

**Honest novelty statement.** The idea is not new to concept bottlenecks: LaBo (CVPR 2023) selects concepts by discriminability and coverage, and mRMR (TPAMI 2005) is classic feature selection. The contribution here is (i) adapting it to LF-CBM's pipeline, (ii) measuring redundancy in image-activation space, and (iii) evaluating at matched budgets with controls and a human study. Frame it as a *systematic extension and evaluation*, not as a new algorithm.

## 3. Metrics (your list → operational definitions)

| Your metric | Definition | Independent of what the filter optimises? |
|---|---|---|
| Classification accuracy | `acc_val` from the final-layer solver | yes |
| Number of concepts | final count after the authors' interpretability cutoff | n/a |
| Concept redundancy | mean nearest-neighbour \|corr\| (activation space) and nearest-neighbour cosine (CLIP text space), on held-out features | activation: partly (same family); **text: yes** |
| Concept relevance | mean η² on held-out features | **no, circular**: the filter selects on η² |
| Interpretability | NEC (non-zero weights per class), share of unused concepts, **blinded human ratings** | yes |

**Do not use η² alone as the success metric for relevance.** The filter maximises it. The independent evidence is the blinded human rating study (`phase2/rating_study.py`) plus the text-space redundancy.

**Why NEC and controls:** VLG-CBM (NeurIPS 2024) shows accuracy alone is a weak CBM metric because even random concept layers can reach high accuracy, and proposes NEC. So every comparison here is at a **matched number of concepts** against **random** and **top-k-by-CLIP-activation** controls. "We removed concepts and accuracy held" is not enough; the filter must beat chance removal.

## 4. Experimental design

| Item | Choice | Reason |
|---|---|---|
| Datasets | **all five**: CUB-200, CIFAR-10, CIFAR-100, Places365, ImageNet | CUB has many descriptive, overlapping concepts (expect the largest effect); CIFAR-10 has few redundant concepts (expect a small one, itself a finding); CIFAR-100 (892 concepts) and the two large sets test whether the filter scales |
| Methods | baseline (`all`), ours (`greedy`), `random`, `clip_topk` | matched-budget controls |
| Budgets | 0.75 / 0.50 / 0.25 of concepts after `clip_cutoff` (large sets: 0.50), **fixed in advance** | avoids test-set tuning (see §5) |
| Seeds | CIFAR-10/100, CUB: 0, 1, 2 (paired across methods). Places365, ImageNet: **seed 0 only, budget 0.5 only** | variance + paired differences where affordable; each large-dataset run repeats projection learning and the sparse solver over 1.3–1.8 M images, so the large sets get 4 runs each (baseline, ours, random, top-k) and **no error bars** |
| Ablations | relevance-only, diversity-only, mean-aggregation (`--ablate`) | shows each part earns its place |
| Human study | 5 raters × 60 kept + 60 removed concepts, blinded | independent relevance evidence |

Run order (reuses your Phase 1 activation caches, so only the first run per dataset is slow):

```powershell
python phase2\test_concept_quality.py ; python tests\test_compat.py   # logic tests
python phase2\smoke_test_glue.py                                       # must say "REAL torch" on your laptop
python phase2\run_phase2.py --dataset cifar10 --seeds 0 --budgets 0.5 --batch_size 64   # first real run
python phase2\run_phase2.py --dataset cifar10  --batch_size 64         # then each dataset with its preset
python phase2\run_phase2.py --dataset cub      --batch_size 64
python phase2\run_phase2.py --dataset cub      --batch_size 64 --ablate
python phase2\run_phase2.py --dataset cifar100 --batch_size 64
python phase2\run_phase2.py --dataset places365 --batch_size 32        # last: see "RAM" below
python phase2\run_phase2.py --dataset imagenet  --batch_size 32
python phase2\aggregate_phase2.py --dataset cub ; python phase2\aggregate_phase2.py --all
python phase2\rating_study.py export --dataset cub --seed 0 --keep 50 --n 60
```

**RAM for Places365 / ImageNet.** The authors' pipeline keeps `target_features` (N×2048) and the CLIP concept scores
(N×K) in memory together: about 33 GB (Places365: 1.80 M × 2,544 concepts) and 35 GB (ImageNet: 1.28 M × 4,751 concepts)
for the baseline, before overhead. Filtered runs keep fewer concepts and need less. On a 16 GB laptop this needs a very
large page file (slow) or may not finish. Order the datasets so the three small ones are finished first; treat the two
large ones as best-effort and report honestly if RAM or time stops them. A documented fallback is training on a fixed
random subset of training images, but that departs from the authors' protocol and must be labelled as such.

**First sanity check (do this before anything else):** `baseline_k100_s0` from the new trainer must reproduce your Phase 1 accuracy for the same dataset and seed (within noise). If not, the modified trainer changed the pipeline and nothing after it is trustworthy.

## 5. Threats to validity (put these in the paper)

1. **Val = test.** For CIFAR/CUB the authors' "val" split is the test split, and their pipeline already uses it for early stopping and the interpretability cutoff. We keep that protocol for comparability, but our filter uses **train data only**, and budgets are fixed in advance.
2. **η² has a small positive bias** (about (C−1)/(N−1) under the null, ≈0.03 for CUB). It is the same for every concept, so rankings are unaffected, but absolute values should be read as relative.
3. **The filter works on CLIP scores; the final model uses learned projection neurons.** A concept can look fine in CLIP space and still be learned poorly. The authors' interpretability cutoff runs afterwards, so final counts differ across methods. Report the final counts.
4. **Fixed `lam`.** NEC is not controlled; it is reported. Accuracy at equal NEC would need a λ sweep (stretch goal).
5. **3 seeds** (small datasets) give rough error bars only; **Places365 and ImageNet have one seed and no error bars.** Report paired differences and say so.
5b. **Scale.** For the large sets, concept relevance/redundancy are estimated from ~1.3–1.8 M images but the final model and the metric are
   still evaluated on one validation split each; and RAM may force a documented fallback (see §4).
6. **Row-centring and the 0.8 / 0.9 duplicate thresholds are design choices.** Report the distributions, not only the thresholded fractions.
7. **What has and has not been verified:** the selection/metrics code passes synthetic tests (planted irrelevant and duplicate concepts). The trainer patch was checked by diff (one authors' line changed, 62 added) and by a smoke test with a numpy stand-in for torch. **It has not yet been run against real torch, real CLIP scores, or your data**; no accuracy or quality numbers exist yet.

## 6. Hypotheses, stated before running

- H1: on CUB, ours beats random and top-k at 50% and 25% budgets in accuracy, redundancy, and human-rated relevance.
- H2: on CIFAR-10 the differences are small (few redundant concepts exist).
- H3: ours at 75% matches the baseline's accuracy within seed noise.
- H4 (large sets, single seed): the *direction* of the CUB/CIFAR effects repeats on Places365 and ImageNet. With one seed this is
  suggestive only, and differences smaller than the seed-to-seed spread seen on the small datasets should not be claimed.
- If H1 fails, report it. A filter that does not beat chance removal is a legitimate negative result and the controls are what make it visible.

## 7. Timeline (today is Oct 12; 22 days)

| Dates | Work | Owner (suggested) |
|---|---|---|
| Oct 12–15 | Environment check, tests, sanity check (baseline = Phase 1), first CIFAR-10 run | Member 2 |
| Oct 15–22 | CIFAR-10, CUB (+ablations), CIFAR-100 sweeps in the background | Member 3 + Member 4 |
| Oct 20–Nov 1 | Places365 and ImageNet runs (best-effort, longest, start as early as RAM allows) | Member 4 + Member 5 |
| Oct 19–25 | Literature section, related-work table (LaBo, VLG-CBM, mRMR, LF-CBM) | Member 1 |
| Oct 23–28 | Aggregate, figures, **human rating study (all 5 raters)** | Member 5 + all |
| Oct 26–Nov 1 | Write-up, 1-page gap explanation, `AI_USAGE.md`, README, repo cleanup | Member 5 + Member 1 |
| Nov 2 | Buffer / proofreading / similarity check (<20%) | all |

## 8. References

LaBo, VLG-CBM and mRMR were checked against their abstracts/pages by search while preparing this plan; LF-CBM details come from the paper and repo used in Phase 1. Re-check page numbers and venues before submitting.

- Oikarinen, Das, Nguyen, Weng. *Label-Free Concept Bottleneck Models.* ICLR 2023.
- Yang, Panagopoulou, Zhou, Jin, Callison-Burch, Yatskar. *Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification.* CVPR 2023. arXiv:2211.11158.
- Srivastava, Yan, Weng. *VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance.* NeurIPS 2024. arXiv:2408.01432.
- Peng, Long, Ding. *Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy.* IEEE TPAMI 27(8):1226–1238, 2005.
