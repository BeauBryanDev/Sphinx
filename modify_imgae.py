#!/usr/bin/env python3

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import argparse
import os

"""
Use case : 
    python resize_to_224.py <imagen>
    python resize_to_224.py <imagen> --size 224
    python resize_to_224.py <imagen> --margin 8 --output ./output.png

"""

def detect_background_color(img: np.ndarray) -> tuple:
    """
    It detect the background color by using the px color mean 
    """
    border = np.concatenate([
        img[0,  :].reshape(-1, 3),   #  upper row 
        img[-1, :].reshape(-1, 3),   #  bottom row 
        img[:,  0].reshape(-1, 3),   #  left col -start -
        img[:, -1].reshape(-1, 3),   #  right col  -end-
    ])
    return tuple(int(x) for x in np.median(border, axis=0))


def resize_to_square(img: np.ndarray,
                     target: int = 224,
                     margin: int = 6) -> np.ndarray:
    """
    Scale image up to the largest possible size ( target -2 * margin ) constraint 
    It  aim to keep aspet ration ,  center inner sign 
    """
    h, w      = img.shape[:2]
    max_side  = target - 2 * margin
    scale     = max_side / max(h, w)
    new_w     = int(w * scale)
    new_h     = int(h * scale)

    resized   = cv2.resize(img, (new_w, new_h),
                           interpolation=cv2.INTER_LANCZOS4)

    bg_color  = detect_background_color(img)
    canvas    = np.full((target, target, 3), bg_color, dtype=np.uint8)

    x_off     = (target - new_w) // 2
    y_off     = (target - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Adaptative Resize"
    )
    parser.add_argument("image_path", help="Income Image path")
    parser.add_argument("--size",   type=int, default=224,
                        help="Output square Size (default: 224)")
    parser.add_argument("--margin", type=int, default=6,
                        help="Px Margin around each sign  (default: 6)")
    parser.add_argument("--output", type=str, default=None,
                        help="Ougtput size (default: <nombre>_224.png besides original image )")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"ERROR: No FOund! — {args.image_path}")
        return

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"ERROR: Unable to Open! — {args.image_path}")
        return

    h, w = img.shape[:2]
    result = resize_to_square(img, target=args.size, margin=args.margin)

    if args.output:
        out_path = args.output
    else:
        folder   = os.path.dirname(os.path.abspath(args.image_path))
        name, _  = os.path.splitext(os.path.basename(args.image_path))
        out_path = os.path.join(folder, f"{name}_{args.size}.png")

    cv2.imwrite(out_path, result)
    print(f"  {w}x{h}  →  {args.size}x{args.size}  ✓  {out_path}")



if __name__ == "__main__":
    main()