#!/usr/bin/env python3
"""Rename blobs inside an ncnn .param file.

Only blob names (layer inputs/outputs) are renamed; layer names and weight
keys are untouched, so the matching .bin stays valid.

Example:
    python rename_ncnn_blobs.py model.param out.param \
        --rename input.1=input --rename score.8=score_8
"""
import argparse


def rename_param(text: str, mapping: dict[str, str]) -> tuple[str, set[str]]:
    lines = text.splitlines()
    used: set[str] = set()

    # Header: magic, then "<layer_count> <blob_count>". Layer lines start at 3.
    out = lines[:2]
    for line in lines[2:]:
        tokens = line.split()
        if len(tokens) < 4:
            out.append(line)
            continue
        n_bottom = int(tokens[2])
        n_top = int(tokens[3])
        end = 4 + n_bottom + n_top
        for i in range(4, end):
            if tokens[i] in mapping:
                used.add(tokens[i])
                tokens[i] = mapping[tokens[i]]
        out.append(" ".join(tokens))
    return "\n".join(out) + "\n", used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_param")
    parser.add_argument("output_param")
    parser.add_argument("--rename", action="append", default=[],
                        help="old=new, repeatable")
    args = parser.parse_args()

    mapping: dict[str, str] = {}
    for pair in args.rename:
        old, _, new = pair.partition("=")
        if not new:
            raise SystemExit(f"bad --rename value: {pair}")
        mapping[old] = new

    with open(args.input_param, encoding="utf-8") as f:
        text = f.read()
    result, used = rename_param(text, mapping)
    with open(args.output_param, "w", encoding="utf-8") as f:
        f.write(result)

    missing = set(mapping) - used
    if missing:
        print("WARN: names not found (converter may already use target names): "
              + ", ".join(sorted(missing)))
    print(f"renamed {len(used)} blob kinds -> {args.output_param}")


if __name__ == "__main__":
    main()
