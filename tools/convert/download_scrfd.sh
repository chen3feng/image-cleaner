#!/usr/bin/env bash
# SCRFD 10g face detector from insightface buffalo_l (2021 package).
# 10GMACs, 640 input, 2 anchors/stride, 9 outputs (score/bbox/kps x3 strides).
# Official model zoo: https://github.com/deepinsight/insightface/tree/master/python-package
set -euo pipefail

OUT_DIR="${1:-build}"
mkdir -p "$OUT_DIR"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
GH_PROXY="${GH_PROXY:-}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Prefer gh (authenticated API download); fall back to curl + optional proxy.
if command -v gh >/dev/null; then
    gh release download v0.7 -R deepinsight/insightface \
        -p buffalo_l.zip -D "$TMP" --clobber
else
    curl -L --fail --progress-bar -o "$TMP/buffalo_l.zip" "${GH_PROXY}$URL"
fi
( cd "$TMP" && unzip -q buffalo_l.zip )
cp "$TMP/det_10g.onnx" "$OUT_DIR/scrfd_10g.onnx"

PYTHON="${PYTHON:-python3}"
"$PYTHON" - "$OUT_DIR/scrfd_10g.onnx" <<'PY' || echo "(onnx inspection skipped)"
import sys
import onnx
m = onnx.load(sys.argv[1])
for i in m.graph.input:
    print("SCRFD input:", i.name,
          [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("SCRFD output:", o.name,
          [d.dim_value for d in o.type.tensor_type.shape.dim])
PY
echo "-> $OUT_DIR/scrfd_10g.onnx"
