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
    "cardboard": 180,
    "glass":     84,
    "metal":     130,
    "paper":     34,
    "plastic":   94,
    "trash":     277,
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
def rand_rotate(img: Image.Image, max_angle: int = 20) -> Image.Image:
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def rand_hflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img

def rand_brightness(img: Image.Image, low: float = 0.85, high: float = 1.15) -> Image.Image:
    factor = random.uniform(low, high)
    return ImageEnhance.Brightness(img).enhance(factor)

def rand_contrast(img: Image.Image, low: float = 0.85, high: float = 1.15) -> Image.Image:
    factor = random.uniform(low, high)
    return ImageEnhance.Contrast(img).enhance(factor)

def rand_saturation(img: Image.Image, low: float = 0.85, high: float = 1.15) -> Image.Image:
    factor = random.uniform(low, high)
    return ImageEnhance.Color(img).enhance(factor)

def rand_color_jitter(img: Image.Image) -> Image.Image:
    img = rand_brightness(img, 0.85, 1.15)
    img = rand_contrast(img, 0.85, 1.15)
    img = rand_saturation(img, 0.9, 1.1)
    return img

def rand_gaussian_blur(img: Image.Image, p: float = 0.3, r_low: float = 0.3, r_high: float = 1.2) -> Image.Image:
    if random.random() < p:
        radius = random.uniform(r_low, r_high)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img

def rand_noise(img: Image.Image, p: float = 0.4, sigma_low: float = 5.0, sigma_high: float = 15.0) -> Image.Image:
    if random.random() >= p:
        return img
    arr = np.asarray(to_rgb(img)).astype(np.float32)
    sigma = random.uniform(sigma_low, sigma_high)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")

def rand_slight_crop(img: Image.Image, keep_low: float = 0.90, keep_high: float = 0.98) -> Image.Image:
    """Crop a central-ish region (keep 90–98%), then resize back to FINAL_SIZE."""
    w, h = img.size
    keep = random.uniform(keep_low, keep_high)
    cw, ch = int(w * keep), int(h * keep)
    max_off_w = max(1, w // 20)
    max_off_h = max(1, h // 20)
    left = max(0, (w - cw) // 2 + random.randint(-max_off_w, max_off_w))
    top  = max(0, (h - ch) // 2 + random.randint(-max_off_h, max_off_h))
    left = min(left, w - cw)
    top  = min(top,  h - ch)
    cropped = img.crop((left, top, left + cw, top + ch))
    return resize_to_final(cropped)

def rand_scale_zoom(img: Image.Image, zoom_low: float = 0.9, zoom_high: float = 1.1) -> Image.Image:
    """Zoom in/out then center-crop/pad back to FINAL_SIZE."""
    img = to_rgb(img)
    w, h = img.size
    zoom = random.uniform(zoom_low, zoom_high)
    nw, nh = max(1, int(w * zoom)), max(1, int(h * zoom))
    zoomed = img.resize((nw, nh), resample=Image.BILINEAR)
    # Center-crop or pad to FINAL_SIZE
    zw, zh = zoomed.size
    left = max(0, (zw - FINAL_SIZE[1]) // 2)
    top  = max(0, (zh - FINAL_SIZE[0]) // 2)
    right = min(zw, left + FINAL_SIZE[1])
    bottom = min(zh, top + FINAL_SIZE[0])
    cropped = zoomed.crop((left, top, right, bottom))
    final = Image.new("RGB", (FINAL_SIZE[1], FINAL_SIZE[0]), (0, 0, 0))
    paste_x = (FINAL_SIZE[1] - cropped.size[0]) // 2
    paste_y = (FINAL_SIZE[0] - cropped.size[1]) // 2
    final.paste(cropped, (paste_x, paste_y))
    return final

def _find_perspective_coeffs(src_pts, dst_pts):
    matrix = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([u for (_, _), (u, v) in zip(src_pts, dst_pts)] + [v for (_, _), (u, v) in zip(src_pts, dst_pts)], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()

def rand_perspective(img: Image.Image, jitter: int = 10) -> Image.Image:
    """Warp then resize back to FINAL_SIZE."""
    img = to_rgb(img)
    w, h = img.size
    src = [(0,0), (w,0), (w,h), (0,h)]
    dst = [
        (random.randint(-jitter, jitter), random.randint(-jitter, jitter)),
        (w + random.randint(-jitter, jitter), random.randint(-jitter, jitter)),
        (w + random.randint(-jitter, jitter), h + random.randint(-jitter, jitter)),
        (random.randint(-jitter, jitter), h + random.randint(-jitter, jitter)),
    ]
    coeffs = _find_perspective_coeffs(src, dst)
    warped = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR)
    return resize_to_final(warped)

#Class-specific pipelines
def pipeline_cardboard(img: Image.Image) -> Image.Image:
    img = to_rgb(img)
    img = rand_rotate(img, 20)
    img = rand_hflip(img, 0.5)
    img = rand_color_jitter(img)
    img = rand_slight_crop(img)  # returns FINAL_SIZE
    return resize_to_final(img)

def pipeline_glass(img: Image.Image) -> Image.Image:
    img = to_rgb(img)
    img = rand_rotate(img, 20)
    img = rand_color_jitter(img)
    return resize_to_final(img)

def pipeline_metal(img: Image.Image) -> Image.Image:
    img = to_rgb(img)
    img = rand_rotate(img, 20)
    img = rand_hflip(img, 0.5)
    img = rand_scale_zoom(img)
    img = rand_noise(img, 0.5, 5.0, 12.0)
    return resize_to_final(img)

def pipeline_paper(img: Image.Image) -> Image.Image:
    img = to_rgb(img)
    img = rand_rotate(img, 10)
    img = rand_brightness(img, 0.9, 1.1)
    img = rand_contrast(img, 0.9, 1.1)
    img = rand_slight_crop(img)
    return resize_to_final(img)

def pipeline_plastic(img: Image.Image) -> Image.Image:
    img = to_rgb(img)
    img = rand_rotate(img, 20)
    img = rand_hflip(img, 0.5)
    img = rand_slight_crop(img)
    return resize_to_final(img)

def pipeline_trash(img: Image.Image) -> Image.Image:
    img = to_rgb(img)
    ops = [
        lambda im: rand_rotate(im, 25),
        lambda im: rand_hflip(im, 0.5),
        lambda im: rand_noise(im, 0.6, 6.0, 15.0),
        lambda im: rand_brightness(im, 0.85, 1.15),
        lambda im: rand_contrast(im, 0.85, 1.15),
        lambda im: rand_perspective(im, 12),
        lambda im: rand_scale_zoom(im, 0.9, 1.1),
    ]
    random.shuffle(ops)
    k = random.randint(4, min(6, len(ops)))
    for op in ops[:k]:
        img = op(img)
    return resize_to_final(img)

PIPELINES = {
    "cardboard": pipeline_cardboard,
    "glass":     pipeline_glass,
    "metal":     pipeline_metal,
    "paper":     pipeline_paper,
    "plastic":   pipeline_plastic,
    "trash":     pipeline_trash,
}

def list_class_images(class_dir: Path) -> List[Path]:
    imgs = [p for p in class_dir.iterdir() if is_image(p)]
    imgs.sort(key=lambda x: x.name.lower())
    return imgs

def ensure_unique_name(dst_dir: Path, stem: str, ext: str = ".jpg") -> Path:
    idx = 0
    while True:
        name = f"{stem}_aug{idx:05d}{ext}"
        p = dst_dir / name
        if not p.exists():
            return p
        idx += 1

def augment_class(class_dir: Path, num_to_generate: int, pipeline_fn):
    images = list_class_images(class_dir)
    if len(images) == 0:
        print(f"Warning: no source images in {class_dir}")
        return 0

    generated = 0
    src_index = 0
    while generated < num_to_generate:
        src = images[src_index % len(images)]
        try:
            with Image.open(src) as im:
                im.load()
                per_source = min(MAX_PER_SOURCE, num_to_generate - generated)
                for _ in range(per_source):
                    aug = pipeline_fn(im)
                    aug = resize_to_final(aug)  # enforce final size
                    dst = ensure_unique_name(class_dir, src.stem, ".jpg")
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
    print(f"Final fixed output size: {FINAL_SIZE} (H, W)")

    total_gen = 0
    for cls, count in AUG_COUNTS.items():
        class_dir = TRAIN_DIR / cls
        if not class_dir.exists():
            print(f"Skipping missing class: {cls}")
            continue
        print(f"\n[{cls}] Generating {count} augmented images...")
        g = augment_class(class_dir, count, PIPELINES[cls])
        print(f"[{cls}] Done. Generated = {g}")
        total_gen += g

    print(f"\nAll done. Total augmented images generated: {total_gen}")

if __name__ == "__main__":
    main()
