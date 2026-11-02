# Phase 2 results — cub

## Original LF-CBM vs improved (ours) at pre-registered budgets

| Method | Budget | n | Accuracy (%) | # concepts | NEC | Redundancy (act.) | Redundancy (text) | Relevance | Unused concepts |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.00 | 3 | 74.12 ± 0.04 | 209 ± 2 | 31.07 ± 0.08 | 0.80 ± 0.00 | 0.89 ± 0.00 | 0.67 ± 0.00 | 0.00 ± 0.00 |
| greedy | 0.25 | 3 | 73.45 ± 0.18 | 63 ± 1 | 21.94 ± 0.12 | 0.65 ± 0.00 | 0.84 ± 0.00 | 0.71 ± 0.00 | 0.00 ± 0.00 |
| greedy | 0.50 | 3 | 74.05 ± 0.41 | 120 ± 1 | 27.43 ± 0.33 | 0.73 ± 0.00 | 0.87 ± 0.00 | 0.70 ± 0.00 | 0.00 ± 0.00 |
| greedy | 0.75 | 3 | 73.94 ± 0.07 | 175 ± 1 | 31.01 ± 0.27 | 0.78 ± 0.00 | 0.88 ± 0.00 | 0.69 ± 0.00 | 0.00 ± 0.00 |

## Matched-budget controls

| Method | Budget | n | Accuracy (%) | # concepts | NEC | Redundancy (act.) | Redundancy (text) | Relevance | Unused concepts |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.25 | 3 | 71.97 ± 0.27 | 52 ± 6 | 20.01 ± 1.32 | 0.72 ± 0.02 | 0.86 ± 0.01 | 0.68 ± 0.01 | 0.00 ± 0.00 |
| random | 0.50 | 3 | 73.68 ± 0.23 | 104 ± 3 | 25.12 ± 0.58 | 0.76 ± 0.01 | 0.88 ± 0.00 | 0.68 ± 0.00 | 0.00 ± 0.00 |
| random | 0.75 | 3 | 73.87 ± 0.13 | 159 ± 1 | 28.72 ± 0.33 | 0.78 ± 0.00 | 0.88 ± 0.00 | 0.67 ± 0.00 | 0.00 ± 0.00 |
| clip_topk | 0.25 | 3 | 72.53 ± 0.06 | 64 ± 1 | 22.29 ± 0.12 | 0.77 ± 0.00 | 0.88 ± 0.00 | 0.72 ± 0.00 | 0.00 ± 0.00 |
| clip_topk | 0.50 | 3 | 73.28 ± 0.24 | 114 ± 1 | 27.28 ± 0.35 | 0.78 ± 0.00 | 0.88 ± 0.00 | 0.70 ± 0.00 | 0.00 ± 0.00 |
| clip_topk | 0.75 | 3 | 73.64 ± 0.09 | 165 ± 2 | 29.86 ± 1.16 | 0.80 ± 0.00 | 0.89 ± 0.00 | 0.68 ± 0.00 | 0.00 ± 0.00 |

## Paired differences (same seed, same budget)

**Ours − random**

| Budget | greedy − random (acc. points) | seeds | wins |
|---|---|---|---|
| 0.25 | +1.48 ± 0.43 | 3 | 3/3 |
| 0.50 | +0.37 ± 0.56 | 3 | 2/3 |
| 0.75 | +0.07 ± 0.19 | 3 | 2/3 |

**Ours − top-k by CLIP activation**

| Budget | greedy − clip_topk (acc. points) | seeds | wins |
|---|---|---|---|
| 0.25 | +0.92 ± 0.13 | 3 | 3/3 |
| 0.50 | +0.77 ± 0.47 | 3 | 3/3 |
| 0.75 | +0.31 ± 0.11 | 3 | 3/3 |

## Ablations

| Method | Budget | n | Accuracy (%) | # concepts | NEC | Redundancy (act.) | Redundancy (text) | Relevance | Unused concepts |
|---|---|---|---|---|---|---|---|---|---|
| abl_diversity_only | 0.25 | 3 | 71.32 ± 0.29 | 34 ± 1 | 16.96 ± 0.38 | 0.49 ± 0.00 | 0.82 ± 0.00 | 0.62 ± 0.00 | 0.00 ± 0.00 |
| abl_diversity_only | 0.50 | 3 | 73.51 ± 0.17 | 85 ± 2 | 23.30 ± 0.21 | 0.61 ± 0.00 | 0.85 ± 0.00 | 0.64 ± 0.00 | 0.00 ± 0.00 |
| abl_diversity_only | 0.75 | 3 | 73.68 ± 0.10 | 140 ± 1 | 27.57 ± 0.17 | 0.70 ± 0.00 | 0.87 ± 0.00 | 0.65 ± 0.00 | 0.00 ± 0.00 |
| abl_mean_agg | 0.25 | 3 | 73.11 ± 0.23 | 79 ± 0 | 24.21 ± 0.04 | 0.81 ± 0.00 | 0.88 ± 0.00 | 0.74 ± 0.00 | 0.00 ± 0.00 |
| abl_mean_agg | 0.50 | 3 | 73.77 ± 0.18 | 150 ± 2 | 29.70 ± 0.24 | 0.81 ± 0.00 | 0.89 ± 0.00 | 0.71 ± 0.00 | 0.00 ± 0.00 |
| abl_mean_agg | 0.75 | 3 | 74.01 ± 0.12 | 196 ± 1 | 32.27 ± 0.20 | 0.80 ± 0.00 | 0.89 ± 0.00 | 0.69 ± 0.00 | 0.00 ± 0.00 |
| abl_relevance_only | 0.25 | 3 | 72.98 ± 0.10 | 83 ± 1 | 24.09 ± 0.05 | 0.82 ± 0.00 | 0.89 ± 0.00 | 0.74 ± 0.00 | 0.00 ± 0.00 |
| abl_relevance_only | 0.50 | 3 | 73.70 ± 0.23 | 154 ± 0 | 29.78 ± 0.19 | 0.81 ± 0.00 | 0.89 ± 0.00 | 0.71 ± 0.00 | 0.00 ± 0.00 |
| abl_relevance_only | 0.75 | 3 | 74.12 ± 0.09 | 195 ± 1 | 32.44 ± 0.26 | 0.81 ± 0.00 | 0.89 ± 0.00 | 0.69 ± 0.00 | 0.00 ± 0.00 |


*Redundancy (act.)* / *(text)* = mean over concepts of the highest similarity to any other kept concept, in image-activation / CLIP-text space. *Relevance* = mean η² on held-out features. *NEC* = non-zero final-layer weights per class. Accuracy is on the authors' `val` split (which is the test split for CIFAR/CUB).
