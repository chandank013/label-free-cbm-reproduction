"""
Runtime compatibility layer.

The authors' code in original_lfcbm/ is kept UNMODIFIED (the only edit in that
folder is the dataset download path in data_utils.py). The problems we hit on a
Windows laptop are instead fixed here, at runtime, by replacing a few
functions/arguments in memory before training starts. Nothing on disk in
original_lfcbm/ changes, and everything can be switched off with --no_compat.

What is applied (see docs/WINDOWS_TROUBLESHOOTING.md for the root causes):

  1. dataloader  [Windows only]  pin_memory=False and num_workers=0 for the
                 DataLoaders inside the authors' utils.py. Fixes
                 "CUDA error: resource already mapped" and Windows error 1455
                 ("paging file too small"). Linux/Colab keep the authors' settings.
  2. memory      [all platforms]  utils.save_target_activations and
                 utils.save_clip_image_features are re-defined: identical
                 behaviour and identical saved files, but the output tensor is
                 PRE-ALLOCATED and filled chunk by chunk instead of
                 torch.cat(list), which needs ~2x the final size in RAM at once.
  3. places365   [all platforms]  torchvision's Places365 only checks that its
                 image FOLDER exists, so an interrupted extraction is silently
                 reused. We add a sampled completeness check.
  4. imagenet    sets DATASET_ROOTS['imagenet_train'/'imagenet_val'] from a
                 folder you pass in, so data_utils.py needn't be edited.

Pretrained model weights are untouched: they still download to the default
locations on the C: drive (~/.torch/models, ~/.cache/torch).
"""
import os
import sys

if os.name != "nt":                       # expandable_segments is unsupported on Windows
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IS_WINDOWS = os.name == "nt"


# ----------------------------------------------------------------------------- 1
def make_safe_loader(real_loader):
    """Wrap torch's DataLoader factory. Windows only: no pinned memory, no workers."""
    def safe_loader(*args, **kw):
        if IS_WINDOWS:
            kw["pin_memory"] = False
            kw["num_workers"] = 0
        return real_loader(*args, **kw)
    return safe_loader


# ----------------------------------------------------------------------------- 2
def cat_inplace(chunks):
    """Same result as torch.cat(chunks) but never holds list + result at once.
    Consumes `chunks` (empties the list) so each batch is freed right after copying."""
    import torch
    total = sum(c.shape[0] for c in chunks)
    out = torch.empty((total,) + tuple(chunks[0].shape[1:]), dtype=chunks[0].dtype)
    chunks.reverse()                      # pop() from the end is O(1) and yields the original order
    i = 0
    while chunks:
        c = chunks.pop()
        out[i:i + c.shape[0]] = c
        i += c.shape[0]
        del c
    return out


def make_memory_efficient_saves(utils):
    """Re-implementations of the authors' two save functions (utils.py, upstream
    commit fa2869d) differing ONLY in how the final tensor is assembled."""
    import torch
    from tqdm import tqdm

    def save_target_activations(target_model, dataset, save_name, target_layers=["layer4"], batch_size=1000,
                                device="cuda", pool_mode='avg'):
        utils._make_save_dir(save_name)
        save_names = {}
        for target_layer in target_layers:
            save_names[target_layer] = save_name.format(target_layer)

        if utils._all_saved(save_names):
            return

        all_features = {target_layer: [] for target_layer in target_layers}

        hooks = {}
        for target_layer in target_layers:
            command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
            hooks[target_layer] = eval(command, vars(utils), dict(
                target_model=target_model, all_features=all_features, target_layer=target_layer, pool_mode=pool_mode))

        with torch.no_grad():
            for images, labels in tqdm(utils.DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                features = target_model(images.to(device))

        for target_layer in target_layers:
            torch.save(cat_inplace(all_features[target_layer]), save_names[target_layer])   # <- only change
            hooks[target_layer].remove()
        del all_features
        torch.cuda.empty_cache()
        return

    def save_clip_image_features(model, dataset, save_name, batch_size=1000, device="cuda"):
        utils._make_save_dir(save_name)
        all_features = []

        if os.path.exists(save_name):
            return

        save_dir = save_name[:save_name.rfind("/")]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with torch.no_grad():
            for images, labels in tqdm(utils.DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                features = model.encode_image(images.to(device))
                all_features.append(features.cpu())
        torch.save(cat_inplace(all_features), save_name)                                    # <- only change
        del all_features
        torch.cuda.empty_cache()
        return

    return save_target_activations, save_clip_image_features


# ----------------------------------------------------------------------------- 3
def patch_places365_integrity_check():
    try:
        import random
        import shutil
        from os import path
        from urllib.parse import urljoin
        from torchvision.datasets.places365 import Places365
        from torchvision.datasets.utils import download_and_extract_archive
    except ImportError:
        return False

    def _patched_download_images(self) -> None:
        if path.exists(self.images_dir):
            sample = random.sample(self.imgs, min(50, len(self.imgs))) if len(self.imgs) else []
            missing = [f for f, _ in sample if not path.exists(f)]
            if not missing:
                return
            print(f"[compat] {self.images_dir} exists but {len(missing)}/{len(sample)} sampled files are missing "
                  f"(e.g. {missing[0]}): interrupted download. Removing and re-downloading.")
            shutil.rmtree(self.images_dir, ignore_errors=True)
        file, md5 = self._IMAGES_META[(self.split, self.small)]
        download_and_extract_archive(urljoin(self._BASE_URL, file), self.root, md5=md5)
        if self.split.startswith("train"):
            os.rename(self.images_dir.rsplit("_", 1)[0], self.images_dir)

    Places365.download_images = _patched_download_images
    return True


# ----------------------------------------------------------------------------- 4
def set_imagenet_roots(data_utils, root):
    train, val = os.path.join(root, "train") + "/", os.path.join(root, "val") + "/"
    for p in (train, val):
        if not os.path.isdir(p):
            raise SystemExit(f"ImageNet folder not found: {p}\n"
                             f"Expected ImageFolder layout (one sub-folder per class) under --imagenet_root={root}. "
                             f"See scripts/imagenet_official/ for the extraction helpers.")
    data_utils.DATASET_ROOTS["imagenet_train"] = train
    data_utils.DATASET_ROOTS["imagenet_val"] = val


# ----------------------------------------------------------------------------- entry point
def apply(enabled=True, dataset=None, imagenet_root=None):
    """Call AFTER `os.chdir(original_lfcbm)` and putting original_lfcbm on sys.path."""
    import data_utils
    import utils
    applied = []
    if dataset == "imagenet" and imagenet_root:
        set_imagenet_roots(data_utils, imagenet_root)       # a setting, not a fix: always applied
        applied.append(f"imagenet_root={imagenet_root}")
    if not enabled:
        print("[compat] fixes DISABLED (--no_compat): authors' code runs exactly as published")
        return applied
    utils.DataLoader = make_safe_loader(utils.DataLoader)
    applied.append("dataloader(" + ("windows: pin_memory=False, num_workers=0" if IS_WINDOWS else "no-op on this OS") + ")")
    utils.save_target_activations, utils.save_clip_image_features = make_memory_efficient_saves(utils)
    applied.append("memory-efficient saves")
    if patch_places365_integrity_check():
        applied.append("places365 integrity check")
    print("[compat] applied: " + ", ".join(applied))
    return applied
