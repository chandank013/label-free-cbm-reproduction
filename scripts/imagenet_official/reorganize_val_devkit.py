"""
Reorganizes the official ILSVRC2012_img_val.tar contents (50,000 flat,
unlabeled images) into ImageFolder format, using the ground truth from the
official devkit (ILSVRC2012_devkit_t12.tar.gz).

The first 1000 entries (in file order) of devkit's data/meta.mat are the
1000 leaf classification synsets, with ILSVRC2012_ID running 1..1000 in
that order. data/ILSVRC2012_validation_ground_truth.txt has 50,000 lines;
line i (1-indexed) is the ILSVRC2012_ID label for image
ILSVRC2012_val_{i:08d}.JPEG.

Requires: pip install scipy

Usage:
    python reorganize_val_devkit.py \
        --val_dir D:/downloads/ILSVRC2012_img_val_extracted \
        --devkit_dir D:/downloads/ILSVRC2012_devkit_t12 \
        --output_dir D:/datasets/ImageNet/val
"""
import argparse
import os
import shutil

from tqdm import tqdm

try:
    from scipy.io import loadmat
except ImportError:
    raise SystemExit("This script needs scipy: pip install scipy")


def build_id_to_wnid(meta_mat_path: str):
    meta = loadmat(meta_mat_path, squeeze_me=True)
    synsets = meta["synsets"]
    id_to_wnid = {}
    for entry in synsets[:1000]:
        ilsvrc_id = int(entry["ILSVRC2012_ID"])
        wnid = str(entry["WNID"])
        id_to_wnid[ilsvrc_id] = wnid
    if len(id_to_wnid) != 1000 or set(id_to_wnid.keys()) != set(range(1, 1001)):
        raise ValueError(
            "Expected the first 1000 synsets in meta.mat to have "
            "ILSVRC2012_ID values 1..1000 exactly. Check you're pointing at "
            "the classification devkit, not a detection/localization variant."
        )
    return id_to_wnid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--devkit_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    meta_mat_path = os.path.join(args.devkit_dir, "data", "meta.mat")
    gt_path = os.path.join(args.devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")
    for p in (meta_mat_path, gt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")

    print("Reading meta.mat and building ID -> WNID lookup...")
    id_to_wnid = build_id_to_wnid(meta_mat_path)

    print("Reading ground truth labels...")
    with open(gt_path) as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if len(labels) != 50000:
        print(f"WARNING: expected 50000 ground truth lines, got {len(labels)}.")

    os.makedirs(args.output_dir, exist_ok=True)
    for wnid in set(id_to_wnid.values()):
        os.makedirs(os.path.join(args.output_dir, wnid), exist_ok=True)

    transfer = shutil.copy2 if args.copy else shutil.move
    print(f"{'Copying' if args.copy else 'Moving'} 50,000 images into class folders...")

    missing = 0
    for i, label_id in enumerate(tqdm(labels, desc="val images")):
        img_name = f"ILSVRC2012_val_{i + 1:08d}.JPEG"
        src = os.path.join(args.val_dir, img_name)
        if not os.path.exists(src):
            missing += 1
            continue
        dst = os.path.join(args.output_dir, id_to_wnid[label_id], img_name)
        transfer(src, dst)

    if missing:
        print(f"WARNING: {missing} expected images were not found in --val_dir.")

    n_classes = len(os.listdir(args.output_dir))
    n_images = sum(len(os.listdir(os.path.join(args.output_dir, d))) for d in os.listdir(args.output_dir))
    print(f"Done. {n_classes} class folders, {n_images} images (expect 1000, 50000).")


if __name__ == "__main__":
    main()
