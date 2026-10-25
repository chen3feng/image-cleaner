#!/usr/bin/env python3
"""Export CodeFormer (w=0, adain=True inference branch) as a split ncnn model.

Why split: the codebook assignment (softmax -> argmax -> one-hot -> embedding
lookup) needs a hard max over the 256x1024 token/codebook matrix. ncnn's
Vulkan kernel for a 2D Reduction on that shape returns garbage on Adreno, which
turns the whole model NaN on GPU; the graph otherwise runs fine on Vulkan
(verified: the same model with the assignment replaced by a plain softmax
produces valid output on GPU).

So the export produces two independent TorchScript/ncnn networks:

  codeformer_enc:  image (1,3,512,512) -> logits (1,256,1024), lq (1,256,16,16)
  codeformer_gen:  quant (1,256,16,16) -> image (1,3,512,512)

The exact hard argmax + codebook lookup + AdaIN happen in C++
(native/core/codeformer.cpp), where they are exact and cost microseconds.
The encoder (conv/transformer) and generator (conv) stay on the GPU.

Input/Output are RGB in [-1, 1] (normalization is done by the caller).

Weights: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth

Usage:
    python export_codeformer.py --weights codeformer.pth --out-dir build
"""
import argparse

import os
import sys

import torch
import torch.nn as nn


def build_net(weights: str) -> nn.Module:
    # Export-only vendor package (see vendor/basicsr), avoids the real basicsr
    # package's training-only dependency chain.
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    sys.path.insert(0, vendor)
    from basicsr.archs.codeformer_arch import CodeFormer

    # Mirrors the construction in inference_codeformer.py / arch registry.
    net = CodeFormer(
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=["32", "64", "128", "256"],
    )
    checkpoint = torch.load(weights, map_location="cpu")
    params = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    net.load_state_dict(params)
    net.eval()
    return net


class EncoderNet(nn.Module):
    """encoder -> transformer -> codebook logits; also emits the style feature.

    Returns (logits, lq_feat):
      logits  (1, 256, 1024)   per-token codebook scores
      lq_feat (1, 256, 16, 16) encoder bottleneck used for AdaIN style stats

    The transformer uses nn.MultiheadAttention, which pnnx maps to ncnn's
    MultiHeadAttention layer. That layer is not bit-exact against PyTorch
    (measured ~32% codebook argmax drift on a test face), but the resulting
    restoration is visually indistinguishable. An explicit matmul/softmax
    rewrite was tried and matched PyTorch exactly in TorchScript, yet pnnx
    leaked opaque nn.Linear/nn.LayerNorm nodes for it that ncnn cannot load,
    so the stock module stays.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        net = self.net
        for block in net.encoder.blocks:
            x = block(x)
        lq_feat = x

        pos_emb = net.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        query_emb = net.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        for layer in net.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        logits = net.idx_pred_layer(query_emb).permute(1, 0, 2)
        return logits, lq_feat


class GeneratorNet(nn.Module):
    """AdaIN-normalised codebook feature (1,256,16,16) -> image (1,3,512,512)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, quant):
        x = quant
        for block in self.net.generator.blocks:
            x = block(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="codeformer.pth")
    parser.add_argument("--out-dir", default="build")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    net = build_net(args.weights)
    enc = EncoderNet(net).eval()
    gen = GeneratorNet(net).eval()

    x = torch.zeros(1, 3, 512, 512)
    quant = torch.zeros(1, 256, 16, 16)

    enc_path = os.path.join(args.out_dir, "codeformer_enc.pt")
    torch.jit.trace(enc, x, strict=False).save(enc_path)
    print(f"wrote {enc_path}")

    gen_path = os.path.join(args.out_dir, "codeformer_gen.pt")
    torch.jit.trace(gen, quant, strict=False).save(gen_path)
    print(f"wrote {gen_path}")

    # Sanity check: full reference path vs the split path with exact hardening.
    with torch.no_grad():
        ref = net(x, w=0, adain=True)[0]
        logits, lq = enc(x)
        one_hot = torch.nn.functional.one_hot(
            logits.argmax(dim=2), logits.shape[2]).type_as(lq)
        q = (one_hot @ net.quantize.embedding.weight)
        q = q.view(1, 16, 16, 256).permute(0, 3, 1, 2)
        # AdaIN, matching adaptive_instance_normalization(quant, lq)
        cm = q.mean(dim=(2, 3), keepdim=True)
        cs = q.var(dim=(2, 3), unbiased=False, keepdim=True).add(1e-5).sqrt()
        sm = lq.mean(dim=(2, 3), keepdim=True)
        ss = lq.var(dim=(2, 3), unbiased=False, keepdim=True).add(1e-5).sqrt()
        out = gen((q - cm) / cs * ss + sm)
        diff = (ref - out).abs().max().item()
        print(f"split-vs-reference max abs diff on zero input: {diff:.6f}")


if __name__ == "__main__":
    main()
