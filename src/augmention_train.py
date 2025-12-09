"""Reads images from each class folder under TRAIN_DIR.
- Generates a fixed number of augmented images per class (as specified in AUG_COUNTS).
- Applies class-specific augmentation pipelines.
- Saves outputs into the same class folders with unique names (suffix _augXXXXX).
-Per-class Data Augmentation with fixed output size (keep dataset native size).
- Final size is set to (H, W) = (384, 512) by default (landscape).
- Applies class-specific augmentation and resizes back to FINAL_SIZE before saving."""
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

#This is Mahmouds Path
TRAIN_DIR = Path(r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\data\split\train")

FINAL_SIZE = (384, 512)  # change to (512, 384) if your images are portrait
#Per-class augmentation targets --------
AUG_COUNTS = {
    "cardboard": 280,
    "glass":     184,
    "metal":     230,
    "paper":     134,
    "plastic":   194,
    "trash":     377,
}

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SAVE_QUALITY_JPEG = 95
MAX_PER_SOURCE = 6

random.seed(SEED)
np.random.seed(SEED)

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def resize_to_final(img: Image.Image) -> Image.Image:
    """Resize to FINAL_SIZE regardless of intermediate ops."""
    return img.resize((FINAL_SIZE[1], FINAL_SIZE[0]), resample=Image.BILINEAR)

# ----- Aug ops (use original size, then force FINAL_SIZE at the end) -----
def rand_rotate(img, max_angle=10):
    ang = random.uniform(-max_angle, max_angle)
    return img.rotate(ang, resample=Image.BICUBIC, expand=True)

def rand_hflip(img, p=0.5):
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img

def rand_brightness(img, low=0.9, high=1.1):
    factor = random.uniform(low, high)
    return ImageEnhance.Brightness(img).enhance(factor)

def rand_contrast(img, low=0.9, high=1.1):
    factor = random.uniform(low, high)
    return ImageEnhance.Contrast(img).enhance(factor)

def rand_slight_crop(img, keep_low=0.95, keep_high=0.99):
    w, h = img.size
    keep = random.uniform(keep_low, keep_high)
    cw, ch = int(w * keep), int(h * keep)

    left = (w - cw) // 2 + random.randint(-5, 5)
    top  = (h - ch) // 2 + random.randint(-5, 5)
    left = max(0, min(left, w - cw))
    top  = max(0, min(top, h - ch))

    cropped = img.crop((left, top, left + cw, top + ch))
    return resize_to_final(cropped)

def pipeline_hog_safe(img):
    img = to_rgb(img)
    img = rand_rotate(img, 10)
    img = rand_hflip(img, 0.5)
    img = rand_brightness(img, 0.9, 1.1)
    img = rand_contrast(img, 0.9, 1.1)
    img = rand_slight_crop(img)
    return resize_to_final(img)


# --------- IMAGE LISTING ----------
def list_class_images(class_dir: Path):
    imgs = [p for p in class_dir.iterdir() if is_image(p)]
    imgs.sort(key=lambda p: p.name.lower())
    return imgs

def ensure_unique_name(dst_dir: Path, stem: str) -> Path:
    idx = 0
    while True:
        name = f"{stem}_aug{idx:05d}.jpg"
        p = dst_dir / name
        if not p.exists():
            return p
        idx += 1


# --------- AUGMENTATION FUNCTION ----------
def augment_class(class_dir: Path, num_to_gen: int):
    """Generate `num_to_gen` augmented images for a given class folder."""
    images = list_class_images(class_dir)
    if len(images) == 0:
        print(f" No images found in {class_dir}")
        return 0

    generated = 0
    src_idx = 0

    while generated < num_to_gen:
        src = images[src_idx % len(images)]
        with Image.open(src) as im:
            im.load()

            aug = pipeline_hog_safe(im)
            dst = ensure_unique_name(class_dir, src.stem)
            aug.save(dst, quality=95)

            generated += 1
        src_idx += 1

    return generated




# -------------- MAIN ----------------
def main():
    total = 0
    print(f"Final output size: {FINAL_SIZE}\n")

    for cls, n_aug in AUG_COUNTS.items():
        cls_dir = TRAIN_DIR / cls
        if not cls_dir.exists():
            print(f"Skipping missing folder: {cls}")
            continue

        print(f"[{cls}] Generating +{n_aug} images...")
        g = augment_class(cls_dir, n_aug)
        print(f"Done → Generated: {g}")
        total += g

    print(f"\nAll done. Total generated: {total}")

if __name__ == "__main__":
    main()