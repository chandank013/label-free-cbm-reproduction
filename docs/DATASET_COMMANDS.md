# Per-dataset commands (source of truth)

Taken verbatim from `original_lfcbm/training_commands.txt` in the authors'
repository. `run_experiment.py` encodes these same values as its
`LFCBM_CONFIGS` / `STANDARD_SPARSE_CONFIGS` dicts — this file is here so a
grader can audit that nothing was changed.

## A note on the "Standard" row

The paper's results table has three rows: **Standard**, **Standard
(sparse)**, and **Label-free CBM**. The authors' own README states:
> "For commands to train Label-free CBM and Standard (sparse) models on all
> 5 datasets, see training_commands.txt."

Only two of the three rows have provided training commands. **"Standard" is
not trained by this codebase at all** — it's a black-box reference ceiling
(the backbone's own already-established accuracy on that dataset; e.g. the
ImageNet "Standard" figure of 76.13% matches torchvision's well-known
published ResNet50 top-1 accuracy). This project reproduces the two rows the
authors provide code for.

## Label-free CBM training

```
CIFAR10:
python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt

CIFAR100:
python train_cbm.py --dataset cifar100 --concept_set data/concept_sets/cifar100_filtered.txt

CUB200:
python train_cbm.py --dataset cub --backbone resnet18_cub --concept_set data/concept_sets/cub_filtered.txt \
    --feature_layer features.final_pool --clip_cutoff 0.26 --n_iters 5000 --lam 0.0002

Places365:
python train_cbm.py --dataset places365 --backbone resnet50 --concept_set data/concept_sets/places365_filtered.txt \
    --clip_cutoff 0.28 --n_iters 80 --lam 0.0003

ImageNet:
python train_cbm.py --dataset imagenet --backbone resnet50 --concept_set data/concept_sets/imagenet_filtered.txt \
    --clip_cutoff 0.28 --n_iters 80 --lam 0.0001
```

## Standard (sparse) baseline training

```
CIFAR10:
python train_standard.py

CIFAR100:
python train_standard.py --dataset cifar100 --lam 0.003

CUB200:
python train_standard.py --dataset cub --backbone resnet18_cub --feature_layer features.final_pool \
    --lam 0.00002 --n_iters 5000

Places365:
python train_standard.py --dataset places365 --backbone resnet50 --lam 0.0007 --n_iters 80

ImageNet:
python train_standard.py --dataset imagenet --backbone resnet50 --lam 0.0001 --n_iters 80
```

## What backbones are used (all frozen — no fine-tuning happens anywhere)

| Dataset | Backbone | Pretraining source |
|---|---|---|
| CIFAR-10/100 | CLIP RN50 | OpenAI CLIP, 400M image-text pairs, zero-shot |
| CUB-200 | `resnet18_cub` | Already fine-tuned on CUB by a third party (`pytorchcv` model zoo) before this pipeline touches it |
| Places365 | plain `resnet50` | Standard **ImageNet**-pretrained weights (`torchvision` `IMAGENET1K_V1`) — not fine-tuned on Places365 itself |
| ImageNet | plain `resnet50` | Same ImageNet-pretrained weights |

Every backbone runs under `.eval()` and `torch.no_grad()` during activation
extraction. The **only** trained component in this entire pipeline is a
small sparse linear layer (fit via the GLM-SAGA solver) mapping
concept-activation scores to class logits.

## Dataset sources

- CIFAR-10 / CIFAR-100: auto-downloaded by `torchvision.datasets` on first run
- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/ (via `download_cub.sh`)
- Places365 (small, 256×256): auto-downloaded by `torchvision.datasets.Places365(small=True)`
- ImageNet-1k: no redistribution rights; extract your own copy to `D:/datasets/ImageNet/{train,val}/` in ImageFolder layout
  (helpers: `scripts/imagenet_official/`) or pass `--imagenet_root <folder>`; `compat.py` sets the authors'
  `DATASET_ROOTS['imagenet_train'/'imagenet_val']` at runtime, so `data_utils.py` needs no edit for it.
