import os
import shutil
from pathlib import Path
from PIL import Image
import torchvision.datasets as datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path("C:/XAI-Project/datasets/cache")

def populate_cifar10():
    out_dir = DATA_DIR / "cifar10"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading CIFAR-10 test set...")
    ds = datasets.CIFAR10(root=str(CACHE_DIR), train=False, download=False)
    classes = ds.classes
    saved = set()
    
    for img, target in ds:
        cls_name = classes[target]
        if cls_name not in saved:
            # Resize slightly from 32x32 to 128x128 for clear viewing/upload
            img_resized = img.resize((128, 128), Image.Resampling.NEAREST)
            dest = out_dir / f"{cls_name}.png"
            img_resized.save(dest)
            saved.add(cls_name)
            print(f"  [CIFAR-10] Saved {dest.name}")
        if len(saved) == len(classes):
            break
    print(f"CIFAR-10 done: {len(saved)} classes.")

def populate_cifar100():
    out_dir = DATA_DIR / "cifar100"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading CIFAR-100 test set...")
    ds = datasets.CIFAR100(root=str(CACHE_DIR), train=False, download=False)
    classes = ds.classes
    saved = set()
    
    for img, target in ds:
        cls_name = classes[target].replace(" ", "_")
        if cls_name not in saved:
            img_resized = img.resize((128, 128), Image.Resampling.NEAREST)
            dest = out_dir / f"{cls_name}.png"
            img_resized.save(dest)
            saved.add(cls_name)
        if len(saved) == len(classes):
            break
    print(f"CIFAR-100 done: {len(saved)} classes.")

def populate_cub():
    out_dir = DATA_DIR / "cub"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cub_test = CACHE_DIR / "CUB" / "test"
    if not cub_test.exists():
        print(f"CUB test directory not found at {cub_test}")
        return

    count = 0
    for cls_folder in sorted(cub_test.iterdir()):
        if not cls_folder.is_dir():
            continue
        # Folder format e.g. "017.Cardinal" -> "cardinal.jpg"
        parts = cls_folder.name.split(".", 1)
        raw_name = parts[1] if len(parts) > 1 else parts[0]
        clean_name = raw_name.lower().replace(" ", "_")
        
        # Pick first image
        imgs = list(cls_folder.glob("*.jpg"))
        if imgs:
            src = imgs[0]
            dest = out_dir / f"{clean_name}.jpg"
            shutil.copyfile(src, dest)
            count += 1
            if count <= 10 or count % 50 == 0:
                print(f"  [CUB] Copied {dest.name}")
    print(f"CUB done: {count} bird classes.")

if __name__ == "__main__":
    populate_cifar10()
    populate_cub()
    populate_cifar100()
    print("All test images populated successfully into main data/ folder!")
