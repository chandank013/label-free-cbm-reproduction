# DA351 — Label-Free Concept Bottleneck Models: Replication (Phase 1) and Concept-Quality Extension (Phase 2)

Paper: Oikarinen, Das, Nguyen, Weng. *Label-Free Concept Bottleneck Models.* ICLR 2023 —
https://openreview.net/pdf?id=FlCg47MNvBA · authors' code: https://github.com/Trustworthy-ML-Lab/Label-free-CBM

## What is (and is not) changed in the authors' code

`original_lfcbm/` is the authors' repository at upstream commit `fa2869d`.
**Exactly one file differs, by one repeated edit:**

| File | Edit |
|---|---|
| `original_lfcbm/data_utils.py` | all 8 `os.path.expanduser("~/.cache")` → `"D:/datasets/cache"` (where CIFAR-10/100 and Places365 download) |

Everything else in that folder is byte-identical to upstream. Pretrained **model
weights** are not touched and still go to their default locations on the C: drive.
To use another cache folder, search-and-replace that one string in `data_utils.py`.
Already downloaded datasets to `C:\Users\<you>\.cache`? `scripts\move_cache_to_d.bat preview` shows what would move (datasets only, never model weights), then run it without `preview`.

Problems we hit on a Windows laptop (CUDA/pin-memory crash, Windows paging-file
error, a 14.8 GB RAM spike, a torchvision download bug) are fixed **outside** the
authors' files, at runtime, by `compat.py`. It is applied automatically by both
runners and can be switched off with `--no_compat` (the crashes will then return on
the same hardware). Details: `docs/WINDOWS_TROUBLESHOOTING.md`.

## Layout

```
original_lfcbm/        authors' code (one edit, see above)
compat.py              runtime fixes, applied from outside the authors' files
run_experiment.py      Phase 1 runner (seeds, logging, calls the authors' training code)
collect_results.py     paper-vs-reproduced table
phase2/                Phase 2: concept filter, generated trainer, runner, aggregation, rating study
tests/  phase2/test_*  logic tests (stand-in torch; see "What is verified")
scripts/imagenet_official/   ImageNet extraction helpers
docs/                  plans, troubleshooting, workflow, presentation script
results/               Phase 1 JSONs; results/phase2/ for Phase 2
```

## Setup

```bash
pip install -r requirements.txt          # install a CUDA build of torch first (docs/LOCAL_GPU_SETUP.md)
cd original_lfcbm
bash download_models.sh                  # pretrained backbones      (Windows: Git Bash)
bash download_cub.sh                     # CUB-200 (needs wget; on Windows use curl.exe + tar, see troubleshooting)
bash download_rn18_places.sh
cd ..
```
CIFAR-10/100 and Places365 download automatically into `D:/datasets/cache` on first use.
ImageNet is never auto-downloaded: extract it to `D:/datasets/ImageNet/{train,val}/`
(ImageFolder layout, helpers in `scripts/imagenet_official/`) or pass `--imagenet_root`.

## Phase 1 — reproduce the paper

```bash
python run_experiment.py --dataset cifar10   --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset cifar100  --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset cub       --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset places365 --model lfcbm --seed 0 --batch_size 32
python run_experiment.py --dataset imagenet  --model lfcbm --seed 0 --batch_size 32
# same with --model standard_sparse for the baseline rows; then
python collect_results.py
```
The paper's plain "Standard" row is a black-box reference ceiling; the authors give no
training command for it, so only *Label-free CBM* and *Standard (sparse)* are reproduced.
Accuracy of a run is `acc_val` in `original_lfcbm/saved_models/<run>/metrics.txt`.

<!-- ## Phase 2 — concept quality filtering, all five datasets

Plan, metrics, controls and caveats: `docs/PHASE2_PLAN.md`.

```bash
python phase2/test_concept_quality.py && python tests/test_compat.py && python phase2/smoke_test_glue.py
python phase2/run_phase2.py --dataset cifar10   --batch_size 64     # 30 runs (3 seeds x 10 configs)
python phase2/run_phase2.py --dataset cifar100  --batch_size 64
python phase2/run_phase2.py --dataset cub       --batch_size 64
python phase2/run_phase2.py --dataset places365 --batch_size 32     #  4 runs (1 seed, budget 0.5)
python phase2/run_phase2.py --dataset imagenet  --batch_size 32     #  4 runs (1 seed, budget 0.5)
python phase2/aggregate_phase2.py --dataset cub    # per-dataset tables + curves
python phase2/aggregate_phase2.py --all            # headline table over all five datasets
```
Runs are resumable and reuse the cached activations from Phase 1. -->

### Be realistic about Places365 and ImageNet

LF-CBM keeps whole-train-set matrices in RAM at once. From the code and the datasets'
sizes: `target_features` (N×2048) plus the CLIP concept scores (N×K) are about
**33 GB for Places365** (1.80 M images, 2,544 concepts) and **35 GB for ImageNet** (1.28 M images,
4,751 concepts) for the *baseline*, before any overhead. That is the authors' pipeline, not
our filter (the filtered runs hold fewer concepts and need less). On a 16 GB laptop this means a very large
page file and slow runs, or failure. Check `systeminfo | findstr /C:"Total Physical Memory"` before committing days to it.
Each of those runs also repeats projection learning and the sparse solver over 1.3–1.8 M images.

## What is verified (and what is not)

- Verified here: the selection/metric code on synthetic data with planted junk and duplicate concepts (7 tests);
  the compat layer's logic with a stand-in for torch (4 tests); the generated trainer by diff against the authors' file
  (1 line changed, 62 added) and by executing the inserted blocks with a stand-in for torch.
- **Not verified here:** anything against real PyTorch, real CLIP scores or real datasets (no GPU or datasets in the
  authoring environment). `python phase2/smoke_test_glue.py` must print `REAL torch` on your machine, and the
  Phase 2 baseline run must reproduce your Phase 1 accuracy before you trust any comparison.
- No accuracy or concept-quality numbers are claimed anywhere in this repo; they must come from your runs.

## Reproducibility checklist (course requirements)

- [x] GitHub repo with code · [x] `requirements.txt` · [x] dataset links (`docs/DATASET_COMMANDS.md`)
- [x] run instructions (this file) · [x] random seeds (`--seed`, logged per run) · [ ] `AI_USAGE.md` filled with your real prompts

## Attribution

Modelling code in `original_lfcbm/` is by Oikarinen et al. (ICLR 2023). `phase2/train_cbm_filtered.py` is generated from the
authors' `train_cbm.py` by `phase2/make_filtered_trainer.py` (inserted blocks are marked `PHASE 2`). The concept-selection idea
relates to LaBo (Yang et al., CVPR 2023) and mRMR (Peng et al., TPAMI 2005); NEC is from VLG-CBM (Srivastava et al., NeurIPS 2024).
