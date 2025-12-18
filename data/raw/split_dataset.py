
# -*- coding: utf-8 -*-
"""
Split images from data/raw into data/split (train/val/test) while keeping class distribution.
Paths are fixed for your environment as requested.
"""

import os
import shutil
import random
from pathlib import Path

# ======== Fixed paths as requested =========
SOURCE_DIR = Path(r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\data\raw")
TARGET_DIR = Path(r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\data\split")
# ===========================================

# Split ratios (75% train / 15% val /10% test)
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10

# General settings
SEED = 42            # For reproducibility
MOVE_FILES = False   # False = copy files (recommended), True = move files

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    """Check if a file is an image based on extension."""
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def split_indices(n: int, train_r: float, val_r: float, seed: int):
    """Return shuffled indices for train, val, and test splits."""
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    n_test  = n - n_train - n_val  # Remaining goes to test
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def copy_or_move(src: Path, dst: Path, move_files: bool):
    """Copy or move a file to the destination, creating folders if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Skip if file already exists (avoid overwriting)
        return
    if move_files:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def main():
    # Check if source exists
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source path does not exist: {SOURCE_DIR}")

    # Get class folders inside source
    classes = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found inside {SOURCE_DIR}")

    print("Detected classes:", [c.name for c in classes])

    # Create target folder structure
    for split in ["train", "val", "test"]:
        for cls in classes:
            (TARGET_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

    total_train = total_val = total_test = 0

    for cls in classes:
        # Collect all images inside the class folder (including subfolders)
        images = [p for p in cls.rglob("*") if is_image(p)]
        images.sort(key=lambda p: p.name.lower())
        n = len(images)
        if n == 0:
            print(f"Warning: No images found in class {cls.name}")
            continue

        train_idx, val_idx, test_idx = split_indices(n, TRAIN_RATIO, VAL_RATIO, SEED)

        # Copy or move files to respective folders
        for i in train_idx:
            dst = TARGET_DIR / "train" / cls.name / images[i].name
            copy_or_move(images[i], dst, MOVE_FILES)
        for i in val_idx:
            dst = TARGET_DIR / "val" / cls.name / images[i].name
            copy_or_move(images[i], dst, MOVE_FILES)
        for i in test_idx:
            dst = TARGET_DIR / "test" / cls.name / images[i].name
            copy_or_move(images[i], dst, MOVE_FILES)

        print(f"[{cls.name}] -> train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} | total={n}")
        total_train += len(train_idx); total_val += len(val_idx); total_test += len(test_idx)

    # Summary
    print("\nSummary:")
    print(f"Train={total_train} | Val={total_val} | Test={total_test}")
    print(f"Files saved in: {TARGET_DIR}")
    print("✔ Files were " + ("moved" if MOVE_FILES else "copied") + ".")

if __name__ == "__main__":
    main()
