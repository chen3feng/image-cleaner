#!/usr/bin/env python3
"""Diagnose the CodeFormer encoder: torch vs ONNX vs ncnn.

The ncnn export used to go TorchScript -> pnnx, where nn.MultiheadAttention is
lowered to ncnn's MultiHeadAttention layer. This script measures the codebook
argmax agreement of each conversion against PyTorch on the same input, so we
can tell whether the attention lowering is the source of the distorted faces.

Usage:
    python diag_encoder.py --weights weights/codeformer.pth --out-dir build/diag
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_net(weights: str):
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    sys.path.insert(0, vendor)
    from basicsr.archs.codeformer_arch import CodeFormer

    net = CodeFormer(
        dim_embd=512, n_head=8, n_layers=9, codebook_size=1024,
        latent_size=256, connect_list=["32", "64", "128", "256"])
    ckpt = torch.load(weights, map_location="cpu")
    net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)))
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", default="build/diag")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from export_codeformer import EncoderNet

    net = build_net(args.weights)
    enc = EncoderNet(net).eval()

    # Deterministic, face-like input in [-1, 1].
    g = torch.Generator().manual_seed(1234)
    x = torch.rand(1, 3, 512, 512, generator=g) * 2 - 1

    with torch.no_grad():
        logits_ref, lq_ref = enc(x)
    logits_ref = logits_ref.numpy()
    lq_ref = lq_ref.numpy()
    x.numpy().astype(np.float32).tofile(os.path.join(args.out_dir, "input.bin"))
    logits_ref.astype(np.float32).tofile(os.path.join(args.out_dir, "logits_torch.bin"))
    lq_ref.astype(np.float32).tofile(os.path.join(args.out_dir, "lq_torch.bin"))
    print("torch logits", logits_ref.shape, "lq", lq_ref.shape)

    ts_path = os.path.join(args.out_dir, "enc.pt")
    torch.jit.trace(enc, x, strict=False).save(ts_path)
    print("wrote", ts_path)

    onnx_path = os.path.join(args.out_dir, "enc.onnx")
    torch.onnx.export(
        enc, x, onnx_path, opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["logits", "lq"], dynamo=False)
    print("wrote", onnx_path)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})
    # ORT output order follows output_names.
    logits_onnx = out[0]
    print("onnx logits", logits_onnx.shape)

    def compare(name, a, b):
        a = a.reshape(256, 1024)
        b = b.reshape(256, 1024)
        agree = (a.argmax(1) == b.argmax(1)).mean()
        print(f"{name}: max|diff|={np.abs(a-b).max():.6f}  argmax agree={agree*100:.2f}%")

    compare("onnx vs torch", logits_onnx, logits_ref)

    # The ncnn runner writes logits_ncnn.bin; compare if it is already there.
    p = os.path.join(args.out_dir, "logits_ncnn.bin")
    if os.path.exists(p):
        compare("ncnn vs torch", np.fromfile(p, dtype=np.float32), logits_ref)


if __name__ == "__main__":
    main()
