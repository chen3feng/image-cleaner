#!/usr/bin/env python3
"""Objective comparison of the 256 (ncnn) and 512 (ORT) DDColor models.

Takes a colour photo, converts it to grayscale (what the app receives), runs
both colorizers, and reports the chroma error against the original colours.

Usage:
    python bench_colorize.py --images build/diag/img/p1015.jpg ...
"""
import argparse
import os
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "diag_ncnn_colorize")
PARAM = os.path.join(HERE, "dist", "colorize-ddcolor.ncnn.param")
BIN = os.path.join(HERE, "dist", "colorize-ddcolor.ncnn.bin")
ONNX512 = os.path.join(HERE, "weights", "ddcolor-512-fp16.onnx")


def to_gray_bgr(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def lab_l(img_bgr):
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 0]


def predict_ab_256(gray):
    l = lab_l(gray)
    lr = cv2.resize(l, (256, 256), interpolation=cv2.INTER_LINEAR)
    lab = np.stack([lr, np.zeros_like(lr), np.zeros_like(lr)], -1)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb.transpose(2, 0, 1).astype(np.float32).tofile("build/diag/bc_in.bin")
    t = time.time()
    subprocess.run([RUNNER, PARAM, BIN, "build/diag/bc_in.bin", "build/diag/bc_out.bin"],
                   check=True, stdout=subprocess.DEVNULL)
    dt = time.time() - t
    return np.fromfile("build/diag/bc_out.bin", dtype=np.float32).reshape(2, 256, 256), dt


def predict_ab_512(sess, gray):
    l = lab_l(gray)
    lr = cv2.resize(l, (512, 512), interpolation=cv2.INTER_LINEAR)
    lab = np.stack([lr, np.zeros_like(lr), np.zeros_like(lr)], -1)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    t = time.time()
    ab = sess.run(None, {"input": inp})[0]
    return ab[0], time.time() - t


def recombine(gray, ab):
    h, w = gray.shape[:2]
    ab_up = cv2.resize(ab.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    l = lab_l(gray)
    out = cv2.cvtColor(np.concatenate([l[:, :, None], ab_up], -1), cv2.COLOR_LAB2BGR)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def chroma_error(orig, out):
    """Mean absolute ab error against the original colours, on a 0..100 scale."""
    a = cv2.cvtColor(orig.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[:, :, 1:]
    b = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[:, :, 1:]
    return float(np.abs(a - b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="build/diag/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(ONNX512, so, providers=["CPUExecutionProvider"])

    for path in args.images:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        s = min(1.0, 900 / max(h, w))
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        gray = to_gray_bgr(img)

        ab256, t256 = predict_ab_256(gray)
        ab512, t512 = predict_ab_512(sess, gray)
        out256 = recombine(gray, ab256)
        out512 = recombine(gray, ab512)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out, f"{stem}_gray.png"), gray)
        cv2.imwrite(os.path.join(args.out, f"{stem}_c256.png"), out256)
        cv2.imwrite(os.path.join(args.out, f"{stem}_c512.png"), out512)
        print(f"{stem}: chroma err 256={chroma_error(img, out256):.2f} "
              f"512={chroma_error(img, out512):.2f} | time 256={t256:.2f}s "
              f"512={t512:.2f}s")


if __name__ == "__main__":
    main()
