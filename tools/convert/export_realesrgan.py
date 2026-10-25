#!/usr/bin/env python3
"""Export Real-ESRGAN general x4v3 to ONNX + TorchScript.

Weights (download separately):
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
Architecture source: tools/convert/vendor/basicsr, a minimal export-only copy
of the basicsr arch modules (the full basicsr package pulls training deps).

Input: 1x3xHxW RGB in [0, 1], output 4x H/W. The network is fully
convolutional; arbitrary tile sizes (256x256) are valid at runtime.

Usage:
    python export_realesrgan.py --weights realesr-general-x4v3.pth --out-dir build
"""
import argparse
import os
import sys

import torch
import torch.nn as nn


def build_net(weights: str) -> nn.Module:
    # The export-only vendor package (see vendor/basicsr) avoids the real
    # basicsr package's training-only dependency chain (scipy, lmdb, …).
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    sys.path.insert(0, vendor)
    # realesr-general-x4v3 is SRVGGNetCompact (not RRDBNet): body.N sequential
    # keys in the checkpoint, 32 conv blocks, PReLU, 4x pixel-shuffle tail.
    from basicsr.archs.srvgg_arch import SRVGGNetCompact

    net = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_conv=32, upscale=4, act_type="prelu",
    )
    checkpoint = torch.load(weights, map_location="cpu")
    params = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    net.load_state_dict(params)
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", default="build")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_net(args.weights)
    dummy = torch.randn(1, 3, 256, 256)

    onnx_path = os.path.join(args.out_dir, "realesrgan-x4v3.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"wrote {onnx_path}")

    try:
        import onnx
        import onnxsim
        model_sim, ok = onnxsim.simplify(onnx_path)
        if ok:
            onnx.save(model_sim, onnx_path)
            print("simplified ONNX")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: onnxsim failed ({exc}); keeping the un-simplified graph")

    ts_path = os.path.join(args.out_dir, "realesrgan-x4v3.pt")
    torch.jit.trace(model, dummy, strict=False).save(ts_path)
    print(f"wrote {ts_path}")


if __name__ == "__main__":
    main()
