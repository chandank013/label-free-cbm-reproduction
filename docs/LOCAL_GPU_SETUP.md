# Running locally on a laptop GPU (6-8GB, Blackwell/RTX 50-series)

Local GPU is preferable to free Colab for this project: no session limits,
no re-downloading datasets, no losing cached activations to a disconnect.

## 1. Check driver + CUDA + PyTorch compatibility first

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Treat `nvidia-smi`'s reported VRAM as ground truth over any spec sheet —
actual available VRAM can differ from advertised specs.

If you get `sm_120 is not compatible` or `torch.cuda.is_available()` is
`False`, your PyTorch build predates Blackwell support:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 2. VRAM budgeting

LF-CBM only does **forward passes** through the frozen backbone — never
backprop. 6-8GB is enough, but default batch sizes assume more headroom.
Use `run_experiment.py`'s `--batch_size` flag:

| Dataset | Backbone | Suggested `--batch_size` |
|---|---|---|
| CIFAR-10/100 | CLIP RN50 | 64 |
| CUB-200 | ResNet18 | 64 |
| Places365 | ResNet50, 224×224 | 32 |
| ImageNet | ResNet50, 224×224 | 32 |

Halve further if you hit `CUDA out of memory`.

## 3. Disk space

| Dataset | Disk needed |
|---|---|
| CIFAR-10 + CIFAR-100 | ~0.5 GB |
| CUB-200-2011 | ~1.5 GB |
| Places365 (small, train-standard) | ~25 GB |
| ImageNet (full) | ~150 GB; val-only substitution ~7 GB |
| Cached activations for all of the above | add another ~5-10 GB |

Keep large datasets **outside** your git-tracked project folder — see
`.gitignore` and the README for why.

## 4. System RAM matters too, not just VRAM

Some activation-saving steps briefly hold a large tensor in **system RAM**
(not GPU memory) before writing to disk — this scales with dataset size and
can exceed 16GB for Places365/ImageNet-scale datasets. If you hit a CPU
`RuntimeError: not enough memory` (not a CUDA error), see
`docs/WINDOWS_TROUBLESHOOTING.md` (section 4). This only affects Places365/ImageNet-scale runs, not CIFAR or CUB.

## 4. System RAM matters as much as VRAM

Several LF-CBM stages hold whole-train-set tensors in **system RAM**. For Places365 and ImageNet the baseline needs roughly
33–35 GB at once (see README, "Be realistic"). `compat.py` removes the extra 2× spike during activation saving but not that
requirement. Check yours: `systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"`.
Where data lives: downloads in `D:/datasets/cache`, pretrained weights on C: (default torch folders).
