#!/usr/bin/env python3
"""Compare DDColor preprocessing/tiling variants offline.

Runs the converted ncnn model (through build/diag_ncnn_colorize) on a photo and
writes whole-image and tiled colorizations so the difference can be inspected.

Usage:
    python diag_colorize.py --image build/diag/img/migrant.jpg --out build/diag/out
"""
import argparse
import os
import subprocess

import cv2
import numpy as np

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "build", "diag_ncnn_colorize")
PARAM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dist", "colorize-ddcolor.ncnn.param")
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "dist", "colorize-ddcolor.ncnn.bin")
SIZE = 256


def lab_l(img_bgr: np.ndarray) -> np.ndarray:
    """L channel in OpenCV float Lab units (0..100), as the app computes it."""
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 0]


def predict_ab(l_crop: np.ndarray, workdir: str) -> np.ndarray:
    """l_crop -> ab (2,256,256) from the network."""
    lr = cv2.resize(l_crop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    lab = np.stack([lr, np.zeros_like(lr), np.zeros_like(lr)], axis=-1)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # 0..1, neutral grey
    inp = rgb.transpose(2, 0, 1).astype(np.float32)
    in_path = os.path.join(workdir, "cin.bin")
    out_path = os.path.join(workdir, "cout.bin")
    inp.tofile(in_path)
    subprocess.run([RUNNER, PARAM, BIN, in_path, out_path], check=True,
                   stdout=subprocess.DEVNULL)
    return np.fromfile(out_path, dtype=np.float32).reshape(2, SIZE, SIZE)


def recombine(img_bgr: np.ndarray, ab: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    ab_up = cv2.resize(ab.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    l = lab_l(img_bgr)
    lab = np.concatenate([l[:, :, None], ab_up], axis=-1)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def whole(img_bgr: np.ndarray, workdir: str) -> np.ndarray:
    ab = predict_ab(lab_l(img_bgr), workdir)
    return recombine(img_bgr, ab)


def tiled(img_bgr: np.ndarray, workdir: str, n: int) -> np.ndarray:
    """n x n overlapping crops at native resolution, colour-matched globally.

    Each crop is predicted on its own, so it can pick a different colour for
    the same object. The global 256 prediction provides the overall colour
    tone; each tile's ab is shifted by the difference between the global and
    tile means before blending with a cosine window.
    """
    h, w = img_bgr.shape[:2]
    l_full = lab_l(img_bgr)
    ab_global = predict_ab(l_full, workdir)
    ab_global_up = cv2.resize(ab_global.transpose(1, 2, 0), (w, h),
                              interpolation=cv2.INTER_LINEAR)

    acc = np.zeros((h, w, 2), np.float32)
    weight = np.zeros((h, w), np.float32)
    win1 = np.hanning(SIZE).astype(np.float32)
    win = np.outer(win1, win1)

    for gy in range(n):
        for gx in range(n):
            x0 = round(gx * (w - SIZE) / max(n - 1, 1)) if w > SIZE else 0
            y0 = round(gy * (h - SIZE) / max(n - 1, 1)) if h > SIZE else 0
            x1 = min(x0 + SIZE, w)
            y1 = min(y0 + SIZE, h)
            x0 = max(0, x1 - SIZE)
            y0 = max(0, y1 - SIZE)
            crop_l = l_full[y0:y1, x0:x1]
            ab = predict_ab(crop_l, workdir)
            ab_up = cv2.resize(ab.transpose(1, 2, 0), (x1 - x0, y1 - y0),
                               interpolation=cv2.INTER_LINEAR)
            ref = ab_global_up[y0:y1, x0:x1]
            # Match the tile's mean colour to the global prediction.
            shift = ref.mean(axis=(0, 1)) - ab_up.mean(axis=(0, 1))
            ab_up = ab_up + shift
            wgt = win[: y1 - y0, : x1 - x0]
            acc[y0:y1, x0:x1] += ab_up * wgt[:, :, None]
            weight[y0:y1, x0:x1] += wgt
    weight = np.maximum(weight, 1e-6)
    return recombine(img_bgr, (acc / weight[:, :, None]).transpose(2, 0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="build/diag/out")
    ap.add_argument("--max-side", type=int, default=900)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("cannot read " + args.image)
    # Keep the diagnostic quick: cap the long edge, like the app's input cap.
    h, w = img.shape[:2]
    scale = min(1.0, args.max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_gray.png"), img)
    cv2.imwrite(os.path.join(args.out, f"{stem}_whole.png"), whole(img, args.out))
    for n in (2, 3):
        cv2.imwrite(os.path.join(args.out, f"{stem}_tile{n}.png"),
                    tiled(img, args.out, n))
    print("wrote results to", args.out)


if __name__ == "__main__":
    main()
