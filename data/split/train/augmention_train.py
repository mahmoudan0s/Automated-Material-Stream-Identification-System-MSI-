"""
Controlled Data Augmentation for MSI Project (TRAIN ONLY)

- Ensures EACH class folder reaches exactly 700 images
- Output size: (224, 224) for EfficientNet-B0
- Each image:
    * Always applies geometric transforms
    * Applies 1 to 3 random appearance / lighting transforms
"""

import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------- PATH ----------------
TRAIN_DIR = Path(
    r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\data\split\train"
)

FINAL_SIZE = (512, 384) # (width, height)
TARGET_COUNT = 550

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SAVE_QUALITY_JPEG = 95

random.seed(SEED)
np.random.seed(SEED)

# ---------------- UTILS ----------------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def resize_to_final(img: Image.Image) -> Image.Image:
    return img.resize(FINAL_SIZE, Image.BILINEAR)

# ---------------- BASIC AUG OPS ----------------
def rand_rotate(img, max_angle=10):
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def rand_hflip(img, p=0.5):
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img

def rand_brightness(img, low=0.9, high=1.1):
    return ImageEnhance.Brightness(img).enhance(random.uniform(low, high))

def rand_contrast(img, low=0.9, high=1.1):
    return ImageEnhance.Contrast(img).enhance(random.uniform(low, high))

def rand_slight_crop(img, keep_low=0.95, keep_high=0.99):
    w, h = img.size
    keep = random.uniform(keep_low, keep_high)
    cw, ch = int(w * keep), int(h * keep)

    left = max(0, (w - cw) // 2 + random.randint(-5, 5))
    top  = max(0, (h - ch) // 2 + random.randint(-5, 5))

    return img.crop((left, top, left + cw, top + ch))

# ---------------- APPEARANCE / LIGHTING OPS ----------------
def rand_color_jitter(img, low=0.95, high=1.05):
    r, g, b = img.split()
    r = ImageEnhance.Brightness(r).enhance(random.uniform(low, high))
    g = ImageEnhance.Brightness(g).enhance(random.uniform(low, high))
    b = ImageEnhance.Brightness(b).enhance(random.uniform(low, high))
    return Image.merge("RGB", (r, g, b))

def rand_gamma(img, low=0.9, high=1.1):
    gamma = random.uniform(low, high)
    inv = 1.0 / gamma
    table = [int((i / 255.0) ** inv * 255) for i in range(256)]
    return img.point(table * 3)

def rand_blur(img):
    return img.filter(ImageFilter.GaussianBlur(
        radius=random.uniform(0.5, 1.0))
    )

# ---------------- AUGMENTATION PIPELINE ----------------
def augmentation_pipeline(img: Image.Image) -> Image.Image:
    img = to_rgb(img)

    # Mandatory geometric transforms
    img = rand_rotate(img, 10)
    img = rand_hflip(img, 0.5)

    # Optional transforms pool
    optional_ops = [
        lambda x: rand_brightness(x, 0.9, 1.1),
        lambda x: rand_contrast(x, 0.9, 1.1),
        rand_color_jitter,
        rand_gamma,
        rand_blur,
    ]

    # Apply 1–3 random optional transforms
    for op in random.sample(optional_ops, random.randint(1, 3)):
        img = op(img)

    img = rand_slight_crop(img)
    img = resize_to_final(img)
    return img

# ---------------- DATA HANDLING ----------------
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

def augment_class_to_target(class_dir: Path, target_count: int) -> int:
    images = list_class_images(class_dir)
    current = len(images)

    if current >= target_count:
        print(f"[{class_dir.name}] Already has {current} images → skipped")
        return 0

    to_generate = target_count - current
    print(f"[{class_dir.name}] {current} → {target_count} (generate {to_generate})")

    generated = 0
    src_idx = 0

    while generated < to_generate:
        src = images[src_idx % len(images)]
        with Image.open(src) as im:
            aug = augmentation_pipeline(im)
            dst = ensure_unique_name(class_dir, src.stem)
            aug.save(dst, quality=SAVE_QUALITY_JPEG)
            generated += 1
        src_idx += 1

    return generated

# ---------------- MAIN ----------------
def main():
    total = 0
    print(f"Target images per class: {TARGET_COUNT}")
    print(f"Final image size: {FINAL_SIZE}\n")

    for cls_dir in TRAIN_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        total += augment_class_to_target(cls_dir, TARGET_COUNT)

    print(f"\nAll done. Total generated images: {total}")

if __name__ == "__main__":
    main()
