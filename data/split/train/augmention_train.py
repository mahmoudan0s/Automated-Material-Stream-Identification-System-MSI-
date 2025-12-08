"""
Per-class Data Augmentation script for MSI project (train set only).
- Reads images from each class folder under the training direction.
- Generates a fixed number of augmented images per class (as specified in AUG_COUNTS).
- Applies class-specific augmentation pipelines.
- Saves outputs into the same class folders with unique names (suffix _augXXXXX).

using Pillow, numpy
"""
import os
import math
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

#This is Mahmoud path 
TRAIN_DIR = Path(r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\data\split\train")

# Target augmentations sfor classes
AUG_COUNTS = {
    "cardboard": 169,
    "glass":     70,
    "metal":     121,
    "paper":     17,
    "plastic":   80,
    "trash":     280,
}

SEED = 42                       # Reproducibility
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SAVE_QUALITY_JPEG = 95
MAX_PER_SOURCE = 6              # Safety cap per source image per loop
random.seed(SEED)
np.random.seed(SEED)

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def list_class_images(class_dir: Path) -> List[Path]:
    imgs = [p for p in class_dir.iterdir() if is_image(p)]
    imgs.sort(key=lambda x: x.name.lower())
    return imgs

def ensure_unique_name(dst_dir: Path, stem: str, ext: str) -> Path:
    """Generate a unique file name to avoid overwriting."""
    idx = 0
    while True:
        name = f"{stem}_aug{idx:05d}{ext}"
        p = dst_dir / name
        if not p.exists():
            return p
        idx += 1

def to_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is in RGB."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

# Augmentation
def rand_rotate(img: Image.Image, max_angle: int = 20) -> Image.Image:
    """Rotate image by a random angle in [-max_angle, +max_angle]."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def rand_hflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    """Horizontal flip with probability p."""
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img

def rand_brightness(img: Image.Image, low: float = 0.8, high: float = 1.2) -> Image.Image:
    """Random brightness adjustment."""
    factor = random.uniform(low, high)
    return ImageEnhance.Brightness(img).enhance(factor)

def rand_contrast(img: Image.Image, low: float = 0.8, high: float = 1.2) -> Image.Image:
    """Random contrast adjustment."""
    factor = random.uniform(low, high)
    return ImageEnhance.Contrast(img).enhance(factor)

def rand_saturation(img: Image.Image, low: float = 0.8, high: float = 1.2) -> Image.Image:
    """Random saturation (color) adjustment."""
    factor = random.uniform(low, high)
    return ImageEnhance.Color(img).enhance(factor)

def rand_color_jitter(img: Image.Image) -> Image.Image:
    """Apply a small random combination of brightness/contrast/saturation."""
    img = rand_brightness(img, 0.85, 1.15)
    img = rand_contrast(img, 0.85, 1.15)
    img = rand_saturation(img, 0.85, 1.15)
    return img

def rand_gaussian_blur(img: Image.Image, p: float = 0.3, radius_low: float = 0.3, radius_high: float = 1.2) -> Image.Image:
    """Gaussian blur with probability p."""
    if random.random() < p:
        radius = random.uniform(radius_low, radius_high)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img

def rand_noise(img: Image.Image, p: float = 0.4, sigma_low: float = 5.0, sigma_high: float = 15.0) -> Image.Image:
    """Add Gaussian noise with probability p."""
    if random.random() >= p:
        return img
    arr = np.asarray(to_rgb(img)).astype(np.float32)
    sigma = random.uniform(sigma_low, sigma_high)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")

def rand_resize_keep_ratio(img: Image.Image, target_short: int = 300) -> Image.Image:
    """Resize keeping aspect ratio so that the shorter side becomes target_short."""
    w, h = img.size
    if min(w, h) == 0:
        return img
    scale = target_short / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BILINEAR)

def rand_slight_crop(img: Image.Image, out_size: Tuple[int, int] = (256, 256), crop_ratio_low: float = 0.85, crop_ratio_high: float = 0.95) -> Image.Image:
    """Slight random crop (keep 85–95% of the image), then resize to out_size."""
    img = rand_resize_keep_ratio(img, target_short=300)
    w, h = img.size
    keep_ratio = random.uniform(crop_ratio_low, crop_ratio_high)
    cw, ch = int(w * keep_ratio), int(h * keep_ratio)
    # random-ish center crop with small offset
    max_off_w = max(1, w // 20)
    max_off_h = max(1, h // 20)
    left = max(0, (w - cw) // 2 + random.randint(-max_off_w, max_off_w))
    top  = max(0, (h - ch) // 2 + random.randint(-max_off_h, max_off_h))
    left = min(left, w - cw)
    top  = min(top,  h - ch)
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize(out_size, resample=Image.BILINEAR)

def rand_scale_zoom(img: Image.Image, out_size: Tuple[int, int] = (256, 256), zoom_low: float = 0.9, zoom_high: float = 1.1) -> Image.Image:
    """Random zoom in/out, then center crop/resize to out_size."""
    img = to_rgb(img)
    w, h = img.size
    zoom = random.uniform(zoom_low, zoom_high)
    nw, nh = int(w * zoom), int(h * zoom)
    zoomed = img.resize((max(1, nw), max(1, nh)), resample=Image.BILINEAR)
    # Center-crop or pad to out_size
    zw, zh = zoomed.size
    # Crop region
    left = max(0, (zw - out_size[0]) // 2)
    top  = max(0, (zh - out_size[1]) // 2)
    right = min(zw, left + out_size[0])
    bottom = min(zh, top + out_size[1])
    cropped = zoomed.crop((left, top, right, bottom))
    # If smaller than out_size, pad
    final = Image.new("RGB", out_size, (0, 0, 0))
    paste_x = (out_size[0] - cropped.size[0]) // 2
    paste_y = (out_size[1] - cropped.size[1]) // 2
    final.paste(cropped, (paste_x, paste_y))
    return final

def _find_perspective_coeffs(src_pts, dst_pts):
    """Compute perspective transform coefficients for PIL.Image.transform."""
    # Solves linear system for perspective coefficients
    matrix = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([u for (_, _), (u, v) in zip(src_pts, dst_pts)] + [v for (_, _), (u, v) in zip(src_pts, dst_pts)], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()

def rand_perspective(img: Image.Image, out_size: Tuple[int, int] = (256, 256), jitter: int = 10) -> Image.Image:
    """Apply a mild random perspective warp."""
    img = rand_resize_keep_ratio(img, target_short=300).convert("RGB")
    w, h = img.size
    # Source corner points
    src = [(0,0), (w,0), (w,h), (0,h)]
    # Destination points jittered by up to +/- jitter pixels
    dst = [
        (random.randint(-jitter, jitter), random.randint(-jitter, jitter)),
        (w + random.randint(-jitter, jitter), random.randint(-jitter, jitter)),
        (w + random.randint(-jitter, jitter), h + random.randint(-jitter, jitter)),
        (random.randint(-jitter, jitter), h + random.randint(-jitter, jitter)),
    ]
    coeffs = _find_perspective_coeffs(src, dst)
    warped = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR)
    return warped.resize(out_size, resample=Image.BILINEAR)
# ---------------------------------------------------------------

# ----------------- Class-specific pipelines -----------------
def pipeline_cardboard(img: Image.Image) -> Image.Image:
    """Rotation ±20°, Horizontal flip, Color jitter, Slight random crop."""
    img = to_rgb(img)
    img = rand_rotate(img, max_angle=20)
    img = rand_hflip(img, p=0.5)
    img = rand_color_jitter(img)
    img = rand_slight_crop(img, out_size=(256, 256))
    return img

def pipeline_glass(img: Image.Image) -> Image.Image:
    """Rotation + ColorJitter only."""
    img = to_rgb(img)
    img = rand_rotate(img, max_angle=20)
    img = rand_color_jitter(img)
    img = rand_slight_crop(img, out_size=(256, 256))  # optional small crop for uniform output size
    return img

def pipeline_metal(img: Image.Image) -> Image.Image:
    """Use 3–4 ops: Rotation, Flip, Scale/Zoom, Noise."""
    img = to_rgb(img)
    img = rand_rotate(img, max_angle=20)
    img = rand_hflip(img, p=0.5)
    img = rand_scale_zoom(img, out_size=(256, 256), zoom_low=0.9, zoom_high=1.1)
    img = rand_noise(img, p=0.5, sigma_low=5.0, sigma_high=12.0)
    return img

def pipeline_paper(img: Image.Image) -> Image.Image:
    """Any simple augment: small rotation + slight brightness/contrast."""
    img = to_rgb(img)
    img = rand_rotate(img, max_angle=10)
    img = rand_brightness(img, 0.9, 1.1)
    img = rand_contrast(img, 0.9, 1.1)
    img = rand_slight_crop(img, out_size=(256, 256))
    return img

def pipeline_plastic(img: Image.Image) -> Image.Image:
    """Rotation + Flip + Crop."""
    img = to_rgb(img)
    img = rand_rotate(img, max_angle=20)
    img = rand_hflip(img, p=0.5)
    img = rand_slight_crop(img, out_size=(256, 256))
    return img

def pipeline_trash(img: Image.Image) -> Image.Image:
    """Heavy diversity: Rotation ±25°, Flip, Noise, Brightness/Contrast jitter, Perspective, Small zooms."""
    img = to_rgb(img)
    # Randomize order
    ops = []
    ops.append(lambda im: rand_rotate(im, max_angle=25))
    ops.append(lambda im: rand_hflip(im, p=0.5))
    ops.append(lambda im: rand_noise(im, p=0.6, sigma_low=6.0, sigma_high=15.0))
    ops.append(lambda im: rand_brightness(im, 0.85, 1.15))
    ops.append(lambda im: rand_contrast(im, 0.85, 1.15))
    ops.append(lambda im: rand_perspective(im, out_size=(256, 256), jitter=12))
    ops.append(lambda im: rand_scale_zoom(im, out_size=(256, 256), zoom_low=0.9, zoom_high=1.1))
    random.shuffle(ops)
    # Apply 4–6 ops
    k = random.randint(4, min(6, len(ops)))
    for op in ops[:k]:
        img = op(img)
    # Ensure uniform size in case perspective chose different path
    img = img.resize((256, 256), resample=Image.BILINEAR)
    return img
# ------------------------------------------------------------

PIPELINES = {
    "cardboard": pipeline_cardboard,
    "glass":     pipeline_glass,
    "metal":     pipeline_metal,
    "paper":     pipeline_paper,
    "plastic":   pipeline_plastic,
    "trash":     pipeline_trash,
}

# Main augmentation routine 
def augment_class(class_dir: Path, num_to_generate: int, pipeline_fn):
    """Generate num_to_generate augmented images into class_dir using pipeline_fn."""
    images = list_class_images(class_dir)
    src_len = len(images)
    if src_len == 0:
        print(f"Warning: no source images in {class_dir}")
        return 0

    generated = 0
    src_index = 0

    while generated < num_to_generate:
        src = images[src_index % src_len]
        try:
            with Image.open(src) as im:
                im.load()
                # Generate up to MAX_PER_SOURCE variants per source per loop
                per_source = min(MAX_PER_SOURCE, num_to_generate - generated)
                for _ in range(per_source):
                    aug = pipeline_fn(im)
                    # Save as JPEG to keep file sizes reasonable unless original is JPEG
                    ext_to_use = ".jpg"
                    dst = ensure_unique_name(class_dir, stem=src.stem, ext=ext_to_use)
                    aug.save(dst, quality=SAVE_QUALITY_JPEG, optimize=True)
                    generated += 1
                    if generated >= num_to_generate:
                        break
        except Exception as e:
            print(f"Error processing {src}: {e}")
        src_index += 1

    return generated

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

    classes = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found in {TRAIN_DIR}")

    
    missing = [cls for cls in AUG_COUNTS.keys() if not (TRAIN_DIR / cls).exists()]
    if missing:
        print(f"Warning: missing class folders {missing}. They will be skipped.")

    print("Augmentation plan:")
    for cls, cnt in AUG_COUNTS.items():
        print(f"- {cls}: {cnt} images")

    total_generated = 0
    for cls_name, target_num in AUG_COUNTS.items():
        class_dir = TRAIN_DIR / cls_name
        if not class_dir.exists():
            print(f"Skipping absent class: {cls_name}")
            continue
        pipeline_fn = PIPELINES.get(cls_name)
        if pipeline_fn is None:
            print(f"No pipeline defined for class: {cls_name}, skipping.")
            continue
        print(f"\n[{cls_name}] Generating {target_num} augmented images...")
        gen = augment_class(class_dir, target_num, pipeline_fn)
        print(f"[{cls_name}] Done. Generated = {gen}")
        total_generated += gen

    print(f"\nAll done. Total augmented images generated: {total_generated}")

if __name__ == "__main__":
    main()
