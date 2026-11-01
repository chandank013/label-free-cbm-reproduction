# Step-by-Step: Running the LF-CBM Replication Locally

Your laptop is actually the *better* environment for this vs. free Colab —
no 12-hour session limit, no re-downloading Places365 if something interrupts
you, and you control disk space directly. The one thing to get right first is
the PyTorch build: Blackwell-generation laptop GPUs (RTX 50-series, compute
capability `sm_120`) need a PyTorch build compiled with CUDA 12.8+ — older
PyTorch versions fail with a `"CUDA capability sm_120 is not compatible"`
error before training even starts.

---

## Step 0 — Check your driver and actual GPU/VRAM

```bash
nvidia-smi
```
Note the exact GPU name and VRAM shown here — treat this as ground truth over
any spec sheet. In this project's testing, `nvidia-smi` / the CUDA OOM error
reported **6GB total**, not the 8GB some spec pages list for this GPU family,
so budget batch sizes off what your own machine reports, not off assumptions.
Also confirm the driver reports **CUDA 12.8 or newer**; if not, update your
Nvidia driver first (GeForce Experience / nvidia.com/drivers).

## Step 1 — Create a clean environment

```bash
python3 -m venv lfcbm-env
source lfcbm-env/bin/activate        # Windows: lfcbm-env\Scripts\activate
python -m pip install --upgrade pip
```
Use Python 3.10 or 3.11 — the repo's `pytorchcv` and `sentence_transformers`
dependencies are best supported there.

## Step 2 — Install PyTorch with Blackwell (cu128) support

**Do this before `pip install -r requirements.txt`**, so the correct GPU
build gets picked up first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify it actually sees your GPU:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected: `True` and your GPU's name printed. If you already got past this
stage (training actually started and ran to the point of caching CLIP
activations before hitting an OOM error, not an `sm_120` error), your
install is already fine — skip re-installing torch and just fix batch size
instead (Step 4).

## Step 3 — Install the rest of the project dependencies

```bash
cd DA351_Phase1_LFCBM
pip install -r requirements.txt
```
This installs `matplotlib`, `gdown`, `pytorchcv`, `sentence_transformers`,
etc. on top of the PyTorch you already have — pip won't downgrade it as long
as the version constraints in `requirements.txt` are satisfied.

## Step 4 — Set batch size for your actual VRAM

Don't use the paper's default `batch_size=512` — on a 6-8GB laptop GPU it
will OOM. Use the `--batch_size` / `--saga_batch_size` flags already built
into `run_experiment.py`:

| Dataset | Backbone | Recommended `--batch_size` |
|---|---|---|
| CIFAR-10 / CIFAR-100 | CLIP RN50 | 64 (raise to 128/256 if `nvidia-smi` shows headroom while running) |
| CUB-200 | ResNet18 | 64 |
| Places365 | ResNet50, 224×224 | 32 |
| ImageNet (val subset) | ResNet50, 224×224 | 32 |

If you still hit `CUDA out of memory`, halve it again (32 → 16), and add
`--saga_batch_size 64` if the OOM happens specifically during the sparse-layer
fitting stage rather than during activation caching.

## Step 5 — Download pretrained backbones + CUB dataset

```bash
cd original_lfcbm
bash download_models.sh
bash download_cub.sh
bash download_rn18_places.sh
cd ..
```
(On Windows without WSL, run these via Git Bash, or manually follow the
`wget`/`gdown`/`tar` commands inside each `.sh` file — they're only a few lines.)
CIFAR-10/100 and Places365 need **no separate download step** — `torchvision`
auto-downloads them the first time you run the training command for that
dataset (Places365's first run will take a while due to the ~24GB download
happening inline).

## Step 6 — Run CIFAR-10 first (sanity check)

```bash
python run_experiment.py --dataset cifar10 --model lfcbm --seed 0 --batch_size 64
```
If this completes without errors, your local GPU setup is confirmed working
end-to-end. Two known issues already fixed in this project's copy of the
code, so you shouldn't hit them:
- `pin_memory` Windows/CUDA race condition (`cudaErrorAlreadyMapped`) — already
  patched to `pin_memory=False` in `original_lfcbm/utils.py`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — already set
  automatically at the top of `run_experiment.py`.

## Step 7 — Run the rest of the "easy" datasets

```bash
python run_experiment.py --dataset cifar100 --model lfcbm --seed 0 --batch_size 64
python run_experiment.py --dataset cub  --model lfcbm --seed 0 --batch_size 64
```

## Step 8 — Places365 (fully feasible locally, unlike free Colab)

Unlike Colab, you don't have a session timeout — just disk space and
bandwidth. Check you have ~30GB free, then:
```bash
python run_experiment.py --dataset places365 --model lfcbm --seed 0 --batch_size 32
```
This can take a while depending on your internet connection, but unlike
Colab it won't be wiped if you close your laptop lid — just disable sleep
temporarily so the download/training isn't interrupted mid-run.

## Step 9 — ImageNet

Full ImageNet-1k (138GB, license required from image-net.org) is *possible*
locally if you have the disk space and are willing to register and download
it. If your deadline doesn't allow that, use the validation-set fallback:
1. Get the ImageNet validation set (~6.3GB, 50,000 images) via Kaggle's
   `imagenet-object-localization-challenge` or Hugging Face's `imagenet-1k`
   (val split, requires accepting the dataset license).
2. In `original_lfcbm/data_utils.py`, set
   `DATASET_ROOTS["imagenet_val"]` to the extracted folder path (must be in
   ImageFolder format — one subfolder per class; the raw val set usually
   isn't, so it needs reorganizing first).
3. Run:
```bash
   python run_experiment.py --dataset imagenet --model lfcbm --seed 0 --batch_size 32 --imagenet_subset
```
State this substitution explicitly in your report either way — full
ImageNet or the val-only fallback are both fine, just be accurate about
which one you did.

## Step 10 — Repeat with the baseline model

```bash
python run_experiment.py --dataset cifar10   --model standard_sparse --seed 0 --batch_size 64
python run_experiment.py --dataset cifar100  --model standard_sparse --seed 0 --batch_size 64
python run_experiment.py --dataset cub       --model standard_sparse --seed 0 --batch_size 64
python run_experiment.py --dataset places365 --model standard_sparse --seed 0 --batch_size 32
python run_experiment.py --dataset imagenet  --model standard_sparse --seed 0 --batch_size 32
```

## Step 11 — Evaluate

```bash
cd original_lfcbm
jupyter notebook evaluate_cbm.ipynb
```
Point it at each trained model in `saved_models/`, run all cells, read off
test accuracy, and fill in `"reproduced_accuracy"` in the matching
`results/<model>_<dataset>_seed0.json`.

## Step 12 — Generate the comparison table and push to GitHub

```bash
cd ..
python collect_results.py
git init && git add . && git commit -m "Phase 1: LF-CBM replication"
git remote add origin https://github.com/<your-team>/<repo-name>.git
git push -u origin main
```

---

## Local-specific troubleshooting

| Problem | Fix |
|---|---|
| `sm_120 is not compatible with the current PyTorch installation` | You installed a pre-Blackwell-support PyTorch or a CPU-only wheel. Re-run Step 2's exact command, in the venv, before anything else touches `torch` |
| `torch.cuda.is_available()` is `False` | Driver too old for CUDA 12.8 — update your Nvidia driver; confirm no other conda/venv on your PATH is shadowing the one you just set up |
| `CUDA error: resource already mapped` in a pin-memory thread | Already patched in this repo's `original_lfcbm/utils.py` (`pin_memory=False`). If it recurs, also change `num_workers=8` to `num_workers=0` in the same file |
| `CUDA out of memory` | Lower `--batch_size`/`--saga_batch_size` (Step 4); close other GPU-using apps (browser hardware acceleration, other training jobs) |
| Laptop overheats / throttles on long Places365 run | Use a cooling pad, keep it plugged in, set Windows power plan to "High performance," not battery saver |
| `pytorchcv` or `sentence_transformers` install fails | Usually a Python-version mismatch — confirm you're on 3.10/3.11 in the venv, not a system Python |