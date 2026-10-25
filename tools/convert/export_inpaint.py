#!/usr/bin/env python3
"""Prepare the LaMa inpainting model for ncnn.

Source: https://huggingface.co/g-ronimo/lama (`lama_512_fp16.onnx`, Apache-2.0).
This is LaMa's FFC network with two changes that make it portable: the FFTs are
replaced by fixed 64x64 DFT matmuls (ncnn has no FFT op) and every BatchNorm is
re-parameterised to scale/shift (LaMa's running_var reaches ~480k, far beyond
fp16 range). The graph is fixed at 512x512.

Two rewrites are needed before pnnx can emit a loadable ncnn model:

1. `Gather(axis=2, index=0|1)` splits the real/imaginary halves of the
   frequency tensor. ncnn has no Gather layer, so each one becomes
   `Slice + Reshape` (both native ncnn layers).
2. The ONNX is weight-only fp16; pnnx's shape inference rejects the mixed
   fp16/fp32 typing, so the graph is widened to fp32 first.

Input  : [1, 4, 512, 512] - RGB (masked area zeroed) + binary mask, values 0..1
Output : [1, 3, 512, 512] - inpainted RGB, values 0..1

Usage:
    python export_inpaint.py --onnx lama_512_fp16.onnx --out-dir build
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


def widen_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(init).astype("float32"), init.name))
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if (value.type.HasField("tensor_type")
                and value.type.tensor_type.elem_type == TensorProto.FLOAT16):
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    return model


def replace_gathers(model: onnx.ModelProto) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(graph.value_info) + list(graph.input) + list(graph.output)
    }
    inits = {i.name: i for i in graph.initializer}

    created: dict[str, list] = {}
    removed = set()
    for node in list(graph.node):
        if node.op_type != "Gather":
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        idx = inits.get(node.input[1])
        if idx is None:
            raise SystemExit(f"non-constant Gather index on {node.name}")
        k = int(numpy_helper.to_array(idx).reshape(-1)[0])
        out_shape = shapes[node.output[0]]

        graph.initializer.extend([
            helper.make_tensor(node.name + "_st", TensorProto.INT64, [1], [k]),
            helper.make_tensor(node.name + "_en", TensorProto.INT64, [1], [k + 1]),
            helper.make_tensor(node.name + "_ax", TensorProto.INT64, [1], [axis]),
            helper.make_tensor(node.name + "_shape", TensorProto.INT64,
                               [len(out_shape)], out_shape),
        ])
        tmp = node.output[0] + "_sliced"
        created[node.name] = [
            helper.make_node(
                "Slice",
                [node.input[0], node.name + "_st", node.name + "_en", node.name + "_ax"],
                [tmp], name=node.name + "_slice"),
            helper.make_node("Reshape", [tmp, node.name + "_shape"],
                             [node.output[0]], name=node.name + "_reshape"),
        ]
        removed.add(node.name)

    nodes = []
    for node in graph.node:
        if node.name in removed:
            nodes.extend(created[node.name])
        else:
            nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(nodes)
    print(f"replaced {len(removed)} Gather nodes")
    return model


def self_check(original_path: str, fixed_path: str) -> None:
    """The rewrites must not change the model's output."""
    rng = np.random.default_rng(0)
    x = rng.random((1, 4, 512, 512), dtype=np.float32)
    # The image channels are 0..1 and the mask is binary.
    x[:, 3] = (x[:, 3] > 0.5).astype(np.float32)
    ref = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"]).run(
        None, {"input": x})[0]
    got = ort.InferenceSession(fixed_path, providers=["CPUExecutionProvider"]).run(
        None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"rewrite self-check: max |original - rewritten| = {diff:.6f}")
    if diff > 0.02:  # fp16 weights vs fp32 reference, so allow a little slack
        raise SystemExit("rewrite changed the model output, aborting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="lama_512_fp16.onnx")
    parser.add_argument("--out-dir", default="build")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "_lama_base.onnx")
    onnx.save(widen_to_fp32(onnx.load(args.onnx)), base)

    model = replace_gathers(onnx.load(base))
    onnx.checker.check_model(model)
    out_path = os.path.join(args.out_dir, "inpaint.onnx")
    onnx.save(model, out_path)
    print(f"wrote {out_path}")

    if not args.skip_check:
        self_check(base, out_path)
    os.remove(base)


if __name__ == "__main__":
    main()
