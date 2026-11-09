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

A third rewrite is needed for speed, not correctness: the DFT matrices are
expressed as 432 MatMuls, and ncnn's MatMul has no Vulkan implementation, so
every one of them copies its input to the CPU and back (288 of them are in the
innermost loop of every FFC block). Each MatMul is the same arithmetic as a
1x1 convolution once the contracted axis is moved into the channel position,
and convolution does run on Vulkan; `matmul_to_conv` does that swap.

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


def matmul_to_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """Turn the DFT MatMuls into 1x1 convolutions.

    A MatMul whose operand is constant is a linear map over one axis. Moving
    that axis into the channel position turns it into a 1x1 convolution, which
    is the same arithmetic but runs on ncnn's Vulkan backend. The LaMa graph
    has 432 of these, 288 of them inside the FFC frequency branch, and ncnn
    falls back to the CPU for every one.
    """
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(graph.value_info) + list(graph.input) + list(graph.output)
    }
    inits = {i.name: i for i in graph.initializer}

    new_nodes = []
    new_inits = []
    converted = 0
    for node in graph.node:
        if node.op_type != "MatMul":
            new_nodes.append(node)
            continue
        a, b = node.input
        if (a in inits) == (b in inits):
            raise SystemExit(f"unexpected MatMul operands on {node.name}")

        if b in inits:
            # out[..., n] = sum_k a[..., k] * b[k, n]; contract the last axis.
            a_shape = shapes[a]
            b_arr = numpy_helper.to_array(inits[b]).astype("float32")
            if len(a_shape) != 4 or b_arr.ndim != 2:
                raise SystemExit(f"unsupported MatMul {node.name}: "
                                 f"{a_shape} x {list(b_arr.shape)}")
            n, k = b_arr.shape[1], b_arr.shape[0]
            # Conv weights are [out, in, kh, kw]; b is [in, out].
            weight = b_arr.transpose(1, 0).reshape(n, k, 1, 1)
            data, perm_in, perm_out = a, [0, 3, 1, 2], [0, 2, 3, 1]
        else:
            # out[..., i, j] = sum_k a[i, k] * b[..., k, j]; the constant sits
            # on the left and contracts the second spatial axis.
            b_shape = shapes[b]
            a_arr = numpy_helper.to_array(inits[a]).astype("float32")
            if a_arr.ndim != 2 or len(b_shape) != 4:
                raise SystemExit(f"unsupported MatMul {node.name}: "
                                 f"{list(a_arr.shape)} x {b_shape}")
            k = a_arr.shape[0]
            weight = a_arr.reshape(k, k, 1, 1)
            data, perm_in, perm_out = b, [0, 2, 1, 3], [0, 2, 1, 3]

        wname = node.name + "_w"
        new_inits.append(numpy_helper.from_array(weight, wname))
        t1 = node.output[0] + "_t1"
        conv = node.output[0] + "_conv"
        new_nodes.extend([
            helper.make_node("Transpose", [data], [t1],
                             name=node.name + "_t1", perm=perm_in),
            helper.make_node("Conv", [t1, wname], [conv],
                             name=node.name + "_conv", pads=[0, 0, 0, 0],
                             strides=[1, 1]),
            helper.make_node("Transpose", [conv], [node.output[0]],
                             name=node.name + "_t2", perm=perm_out),
        ])
        converted += 1

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)

    # The DFT matrices are dead once every MatMul that used them is gone.
    used = {inp for node in graph.node for inp in node.input}
    used.update(v.name for v in graph.output)
    kept = [i for i in graph.initializer if i.name in used]
    if len(kept) != len(graph.initializer):
        graph.ClearField("initializer")
        graph.initializer.extend(kept)
    print(f"rewrote {converted} MatMul nodes as 1x1 convolutions")
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
    model = matmul_to_conv(model)
    onnx.checker.check_model(model)
    out_path = os.path.join(args.out_dir, "inpaint.onnx")
    onnx.save(model, out_path)
    print(f"wrote {out_path}")

    if not args.skip_check:
        self_check(base, out_path)
    os.remove(base)


if __name__ == "__main__":
    main()
