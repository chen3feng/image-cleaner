#!/usr/bin/env python3
"""Prepare the DDColor (ICCV 2023) colorization model for ncnn.

Source: https://huggingface.co/wavespeed/image-colorizer (Apache-2.0 weights,
exported from https://github.com/piddnad/DDColor). The published ONNX is fp16
with a 256x256 input and predicts the Lab ab channels at full resolution.

Two graph fixes are required before pnnx will produce a loadable ncnn model:

1. ncnn's Vulkan/CPU Einsum lowering for `bqc,bchw->bqhw` writes outside its
   buffers (segfault on device). Since B is already laid out as (b, c, h*w),
   the operation is a plain MatMul: A(b,q,c) @ B(b,c,hw) -> (b,q,hw), then a
   reshape back to (b,q,h,w).
2. The ONNX is fp16 and pnnx's shape inference rejects mixed fp16/fp32 typing;
   everything is widened to fp32 first. That also suits ncnn: running DDColor
   with fp16 storage shifts the chroma noticeably (mean ab error 2.2 vs 0.13),
   so it runs as an fp32 CPU net (≈0.8 s for 256x256 on a phone).

Usage:
    python export_colorize.py --onnx ddcolor-fp16.onnx --out-dir build
    pnnx build/colorize.onnx inputshape=[1,3,256,256]
"""

import argparse
import os

import onnx
import onnxruntime as ort
import numpy as np
from onnx import TensorProto, helper, numpy_helper


def widen_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            init.CopyFrom(
                numpy_helper.from_array(numpy_helper.to_array(init).astype("float32"), init.name))
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if (value.type.HasField("tensor_type")
                and value.type.tensor_type.elem_type == TensorProto.FLOAT16):
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    return model


def replace_einsum(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    targets = [n for n in graph.node if n.op_type == "Einsum"]
    if not targets:
        return model
    if len(targets) > 1:
        raise SystemExit(f"unexpected number of Einsum nodes: {len(targets)}")

    node = targets[0]
    equation = next(a.s for a in node.attribute if a.name == "equation")
    if equation != b"bqc,bchw->bqhw":
        raise SystemExit(f"unexpected Einsum equation: {equation}")
    a, b = node.input
    out = node.output[0]

    graph.initializer.extend([
        helper.make_tensor("e_bmat_shape", TensorProto.INT64, [3], [1, 256, 256 * 256]),
        helper.make_tensor("e_out_shape", TensorProto.INT64, [4], [1, 100, 256, 256]),
    ])
    new_nodes = [
        helper.make_node("Reshape", [b, "e_bmat_shape"], ["e_bmat"], name="e_bmat"),
        helper.make_node("MatMul", [a, "e_bmat"], ["e_mm"], name="e_mm"),
        helper.make_node("Reshape", ["e_mm", "e_out_shape"], [out], name="e_out"),
    ]
    idx = list(graph.node).index(node)
    nodes = list(graph.node)[:idx] + new_nodes + list(graph.node)[idx + 1:]
    graph.ClearField("node")
    graph.node.extend(nodes)
    return model


def self_check(original_path: str, fixed_path: str, out_dir: str) -> None:
    """The rewrite must be numerically identical to the source graph."""
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 256, 256), dtype=np.float32)
    ref = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"]).run(
        None, {"input": x})[0]
    got = ort.InferenceSession(fixed_path, providers=["CPUExecutionProvider"]).run(
        None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"rewrite self-check: max |original - rewritten| = {diff:.6f}")
    if diff > 1e-3:
        raise SystemExit("rewrite changed the model output, aborting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="ddcolor-fp16.onnx")
    parser.add_argument("--out-dir", default="build")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = widen_to_fp32(onnx.load(args.onnx))
    base = os.path.join(args.out_dir, "_ddcolor_base.onnx")
    onnx.save(model, base)

    # DDColor's ONNX declares its input as "input"; after pnnx the blobs are
    # renamed to input/output to match the manifest contract.
    fixed = replace_einsum(model)
    onnx.checker.check_model(fixed)
    out_path = os.path.join(args.out_dir, "colorize.onnx")
    onnx.save(fixed, out_path)
    print(f"wrote {out_path}")

    if not args.skip_check:
        self_check(base, out_path, args.out_dir)
    os.remove(base)


if __name__ == "__main__":
    main()
