"""
Extracts the official ILSVRC2012_img_train.tar into ImageFolder format
(one subfolder per class, images directly inside).

The official train archive is NOT a flat folder of images -- it's a single
tar containing 1000 smaller per-class tars (n01440764.tar, ...), each with
that class's JPEGs. Extract twice: outer tar, then each inner tar.

Usage:
    python extract_train_tars.py \
        --train_tar D:/downloads/ILSVRC2012_img_train.tar \
        --output_dir D:/datasets/ImageNet/train
"""
import argparse
import os
import tarfile

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tar", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tmp_dir", default=None)
    ap.add_argument("--keep_inner_tars", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = args.tmp_dir or os.path.join(args.output_dir, "_inner_tars_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Step 1/2: extracting outer tar -> {tmp_dir}")
    with tarfile.open(args.train_tar) as outer:
        members = outer.getmembers()
        for m in tqdm(members, desc="outer tar"):
            outer.extract(m, path=tmp_dir)

    inner_tars = [f for f in os.listdir(tmp_dir) if f.endswith(".tar")]
    print(f"Found {len(inner_tars)} per-class tars (expect 1000).")

    print("Step 2/2: extracting each class's images into its own folder")
    for fname in tqdm(inner_tars, desc="class tars"):
        wnid = fname[:-4]
        class_dir = os.path.join(args.output_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)
        inner_path = os.path.join(tmp_dir, fname)
        with tarfile.open(inner_path) as inner:
            inner.extractall(path=class_dir)
        if not args.keep_inner_tars:
            os.remove(inner_path)

    if not args.keep_inner_tars:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    n_classes = len(os.listdir(args.output_dir))
    print(f"Done. {args.output_dir} has {n_classes} class subfolders (expect 1000).")


if __name__ == "__main__":
    main()
