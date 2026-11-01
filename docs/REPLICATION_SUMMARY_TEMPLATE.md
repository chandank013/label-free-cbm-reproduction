# Replication Summary — Label-Free Concept Bottleneck Models (ICLR 2023)

**Team**: [names, roll numbers] · **Paper**: Oikarinen et al., ICLR 2023 ·
**Code base**: github.com/Trustworthy-ML-Lab/Label-free-CBM

## 1. What we reproduced
We reproduced the Label-free CBM and Standard (sparse) accuracy results
(paper's main results table) on [list datasets you completed: CIFAR-10,
CIFAR-100, CUB-200, and — Places365 / ImageNet at reduced scale, if applicable].

## 2. Setup
- Compute: Google Colab (free tier, T4 GPU)
- Backbones: CLIP RN50 (CIFAR), ResNet18 fine-tuned on CUB, ResNet50 (Places365/ImageNet) — all pretrained, frozen, as specified by the authors
- Seed: 0 (fixed via `torch`/`numpy`/`random`), logged per run in `results/*.json`
- Any deviations from the original setup: [state Places365/ImageNet substitutions here explicitly, if used]

## 3. Results

| Model | Dataset | Paper Reported (%) | Reproduced (%) | Difference | Seed | Runtime (s) |
|---|---|---|---|---|---|---|
| lfcbm | cifar10 | 86.37 | 86.44 | +0.07 | 0 | 594 |
| lfcbm | cifar100 | 65.27 | 65.33 | +0.06 | 0 | 719 |
| lfcbm | cub | 74.59 | 74.16 | -0.43 | 0 | 562 |
| lfcbm | places365 | � | *not run* | � | � | � |
| lfcbm | imagenet | � | *not run* | � | � | � |
| standard_sparse | cifar10 | 82.96 | 82.96 | +0.00 | 0 | 28 |
| standard_sparse | cifar100 | 58.34 | 58.34 | +0.00 | 0 | 136 |
| standard_sparse | cub | 75.96 | 75.92 | -0.04 | 0 | 176 |
| standard_sparse | places365 | � | *not run* | � | � | � |
| standard_sparse | imagenet | � | *not run* | � | � | � |


## 4. Discussion
- Where results matched closely and why (frozen backbone + deterministic
  sparse-layer solver means variance across seeds should be small).
- Where results diverged and likely causes (e.g. reduced ImageNet subset size,
  Places365 val-only substitution, minor library version differences).
- Qualitative check: [1-2 example images with concept-based explanations from
  `evaluate_cbm.ipynb`'s bar plots — do they look sensible/interpretable?]

## 5. Effort breakdown
[Which teammate ran which dataset / did evaluation / wrote this summary —
this doubles as your contribution log.]
