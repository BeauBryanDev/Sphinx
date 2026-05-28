#!/usr/bin/env python3

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import argparse
import os
import random

#  Seed reproducible 
random.seed(None)  
np.random.seed(None)


# Augmentation primitives

def apply_brightness_contrast(img: np.ndarray) -> np.ndarray:
    """Change brightness and contrast moderately."""
    alpha = random.uniform(0.88, 1.32)   # contraste  ±12%
    beta  = random.randint(-15, 15)      # brillo     ±15 puntos
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_rotation(img: np.ndarray) -> np.ndarray:
    """Rotate very slightly ±3° with reflection border fill."""
    angle = random.uniform(-3.0, 3.0)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Soft blur — simulates edge erosion in stone."""
    if random.random() < 0.45:
        return cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.7))
    return img


def apply_gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Moderate gaussian noise — simulates stone grain."""
    if random.random() < 0.65:
        sigma = random.uniform(3.0, 8.0)   
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def apply_salt_pepper(img: np.ndarray) -> np.ndarray:
    """
    Salt & pepper very light     
    it simulates punctual imperfections of the material.
    """
    if random.random() < 0.60:
        out  = img.copy()
        prob = random.uniform(0.0005, 0.008)   
        num_salt = int(prob * img.size * 0.5)
        if num_salt > 0:
            ys = np.random.randint(0, img.shape[0], num_salt)
            xs = np.random.randint(0, img.shape[1], num_salt)
            out[ys, xs] = 255

        num_pepper = int(prob * img.size * 0.5)
        if num_pepper > 0:
            ys = np.random.randint(0, img.shape[0], num_pepper)
            xs = np.random.randint(0, img.shape[1], num_pepper)
            out[ys, xs] = 0

        return out
    return img


def apply_erosion(img: np.ndarray) -> np.ndarray:
    """
    Light Erosion 
    it simulates superficial wear of the carving in stone.
    """
    if random.random() < 0.45:
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(img, kernel, iterations=1)
    return img


def apply_shadow(img: np.ndarray) -> np.ndarray:
    """
    Soft lateral shadow 
    it simulates lateral lighting of museum / tomb.
    """
    if random.random() < 0.45:
        h, w   = img.shape[:2]
        shadow = np.zeros_like(img)
        x      = random.randint(0, w // 2)
        cv2.rectangle(shadow, (x, 0), (w, h), (20, 20, 20), -1)
        alpha  = random.uniform(0.06, 0.24)   # muy sutil
        return cv2.addWeighted(img, 1.0, shadow, alpha, 0)
    return img


def apply_elastic_distortion(img: np.ndarray) -> np.ndarray:
    """
    Elastic  distortion very soft — it simulates micro-irregularities of the carving.
    alpha low (5-9) + sigma high (4-6) = soft global deformation.
    """
    if random.random() < 0.50:
        h, w  = img.shape[:2]
        alpha = random.uniform(5.0, 9.0)
        sigma = random.uniform(4.0, 6.0)
        dx    = cv2.GaussianBlur(
                    (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                    (0, 0), sigma) * alpha
        dy    = cv2.GaussianBlur(
                    (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                    (0, 0), sigma) * alpha
        x, y  = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        return cv2.remap(img, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img


def apply_stone_texture(img: np.ndarray) -> np.ndarray:
    """
    Sandstone/limestone texture: low-frequency noise scaled.
    Very light overlay (8%) so as not to hide the sign.
    """
    if random.random() < 0.55:
        h, w   = img.shape[:2]
       
        small  = np.random.randint(0, 25,
                                   (max(1, h // 6), max(1, w // 6)),
                                   dtype=np.uint8)
        grain  = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        if len(img.shape) == 3:

            grain = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(img, 0.92, grain, 0.08, 0)

    return img



def augment(img: np.ndarray) -> np.ndarray:
    """
    Augmentation pipeline calibrated for hieroglyphics in stone.
    Order: first geometric transformations, then radiometric.
    """
    out = img.copy()

    # — Geometric —
    out = apply_rotation(out)           # ±3°, siempre
    out = apply_elastic_distortion(out) # 50% prob, muy suave
    out = apply_erosion(out)            # 45% prob

    # — Radiometric —
    out = apply_brightness_contrast(out)  # siempre
    out = apply_gaussian_blur(out)        # 35% prob
    out = apply_gaussian_noise(out)       # 55% prob
    out = apply_salt_pepper(out)          # 40% prob

    #   Texture / lighting
    out = apply_stone_texture(out)        # 55% prob
    out = apply_shadow(out)              # 45% prob

    return out



def main():
    parser = argparse.ArgumentParser(
        description="SphinxEyes"
    )
    parser.add_argument("image_path", help="Path to the original image")
    parser.add_argument("copies",     type=int, help="Number of copies to generate")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"ERROR: Image not found — {args.image_path}")
        return

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"ERROR: Could not open image — {args.image_path}")
        return

    folder        = os.path.dirname(os.path.abspath(args.image_path))
    name, ext     = os.path.splitext(os.path.basename(args.image_path))


    existing = [
        f for f in os.listdir(folder)
        if f.startswith(f"{name}_aug_") and f.endswith(ext)
    ]
    start_idx = len(existing) + 1

    generated = 0
    
    for i in range(start_idx, start_idx + args.copies):
        aug       = augment(img)
        save_path = os.path.join(folder, f"{name}_aug_{i:03d}{ext}")
        cv2.imwrite(save_path, aug, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"  ✓  {save_path}")
        generated += 1

    print(f"\n  {generated} copies generated from '{args.image_path}'")


if __name__ == "__main__":

    main()