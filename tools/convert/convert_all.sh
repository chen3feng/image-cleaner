#!/usr/bin/env bash
# End-to-end model conversion (pnnx based, no ncnn-tools build needed):
#   weights (.pth) -> TorchScript/ONNX -> pnnx -> ncnn fp16
#   SCRFD:        insightface ONNX -> onnxsim(static 640) -> pnnx
#
# Prereqs:
#   1) python venv with requirements.txt active (pass with PYTHON=...)
#   2) weights in tools/convert/weights/:
#        - codeformer.pth
#          https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
#        - realesr-general-x4v3.pth
#          https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
#
# Slow-network knobs:
#   GH_PROXY=https://ghfast.top/https://github.com  (prefix for github URLs)
#   PNNX_VERSION=20261020
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Keep the pnnx release in lockstep with the ncnn runtime version
# (tools/setup_ncnn.sh); mismatched generations can emit params the older
# runtime parses into a null layer and crashes.
PNNX_VERSION="${PNNX_VERSION:-20260526}"

BUILD="$SCRIPT_DIR/build"
DIST="${OUT_DIR:-$SCRIPT_DIR/dist}"
CACHE="$SCRIPT_DIR/.cache"
WEIGHTS="$SCRIPT_DIR/weights"
PYTHON="${PYTHON:-python3}"
GH_PROXY="${GH_PROXY:-}"
mkdir -p "$BUILD" "$DIST" "$CACHE" "$WEIGHTS"

# ---------------- pnnx tool ----------------
OS_TAG="$(uname -s)"
case "$OS_TAG" in
    Darwin) PNNX_PKG="pnnx-${PNNX_VERSION}-macos.zip" ;;
    Linux)  PNNX_PKG="pnnx-${PNNX_VERSION}-linux.zip" ;;
    *) echo "Unsupported OS: $OS_TAG" >&2; exit 1 ;;
esac

fetch() { # url dest
    if [[ ! -f "$2" ]]; then
        echo ">> download $1"
        curl -L --fail --progress-bar -o "$2" "${GH_PROXY}$1"
    fi
}

PNNX_ZIP="$CACHE/$PNNX_PKG"
URL="https://github.com/pnnx/pnnx/releases/download/${PNNX_VERSION}/${PNNX_PKG}"
if command -v gh >/dev/null && [[ ! -f "$PNNX_ZIP" ]]; then
    gh release download "$PNNX_VERSION" -R pnnx/pnnx -p "$PNNX_PKG" \
        -D "$CACHE" --clobber
else
    fetch "$URL" "$PNNX_ZIP"
fi
PNNX_DIR="$CACHE/pnnx"
mkdir -p "$PNNX_DIR"
unzip -qn "$PNNX_ZIP" -d "$PNNX_DIR"
PNNX="$(find "$PNNX_DIR" -name pnnx -type f -perm +111 2>/dev/null | head -1)"
[[ -x "$PNNX" ]] || { echo "pnnx not found in $PNNX_DIR" >&2; exit 1; }

rename() { "$PYTHON" "$SCRIPT_DIR/rename_ncnn_blobs.py" "$@"; }

# pnnx emits <stem>.ncnn.param/.bin into cwd.
run_pnnx() { # in-file inputshape-spec
    local in_file="$1"; local shape="$2"
    ( cd "$BUILD" && "$PNNX" "$in_file" "inputshape=$shape" fp16=1 )
}

# ---------------- SCRFD 10g (ONNX -> static -> pnnx) ----------------
if [[ ! -f "$BUILD/scrfd_10g.onnx" ]]; then
    GH_PROXY="$GH_PROXY" PYTHON="$PYTHON" bash "$SCRIPT_DIR/download_scrfd.sh" "$BUILD"
fi
# insightface ships dynamic [1,3,0,0]; freeze to a static 640x640 graph.
if [[ ! -f "$BUILD/scrfd_10g.static.onnx" ]]; then
    "$PYTHON" -m onnxsim "$BUILD/scrfd_10g.onnx" \
              "$BUILD/scrfd_10g.static.onnx" --overwrite-input-shape 1,3,640,640
fi
run_pnnx scrfd_10g.static.onnx "[1,3,640,640]"
# Blob names depend on the pnnx version; the map below is verified against
# pnnx 20260526 (see tools/convert/README.md): out0..out8 group as
# score/bbox/kps x strides 8/16/32.
rename "$BUILD/scrfd_10g.static.ncnn.param" "$DIST/scrfd_10g.ncnn.param" \
    --rename in0=input \
    --rename out0=score_8  --rename out1=score_16 --rename out2=score_32 \
    --rename out3=bbox_8   --rename out4=bbox_16  --rename out5=bbox_32 \
    --rename out6=kps_8    --rename out7=kps_16   --rename out8=kps_32
cp "$BUILD/scrfd_10g.static.ncnn.bin" "$DIST/scrfd_10g.ncnn.bin"

# ---------------- CodeFormer (TorchScript -> pnnx) ----------------
if [[ -f "$WEIGHTS/codeformer.pth" ]]; then
    if [[ ! -f "$BUILD/codeformer.pt" ]]; then
        "$PYTHON" "$SCRIPT_DIR/export_codeformer.py" \
            --weights "$WEIGHTS/codeformer.pth" --out-dir "$BUILD"
    fi
    run_pnnx codeformer.pt "[1,3,512,512]"
    rename "$BUILD/codeformer.ncnn.param" "$DIST/codeformer-fp16.ncnn.param" \
        --rename in0=input --rename out0=output --rename 1=input
    cp "$BUILD/codeformer.ncnn.bin" "$DIST/codeformer-fp16.ncnn.bin"
else
    echo "SKIP CodeFormer: missing $WEIGHTS/codeformer.pth"
fi

# ---------------- Real-ESRGAN general x4v3 (TorchScript -> pnnx) ----------------
if [[ -f "$WEIGHTS/realesr-general-x4v3.pth" ]]; then
    if [[ ! -f "$BUILD/realesrgan-x4v3.pt" ]]; then
        "$PYTHON" "$SCRIPT_DIR/export_realesrgan.py" \
            --weights "$WEIGHTS/realesr-general-x4v3.pth" --out-dir "$BUILD"
    fi
    run_pnnx realesrgan-x4v3.pt "[1,3,256,256]"
    # pnnx turns '-' in the stem into '_' for its output file names.
    rename "$BUILD/realesrgan_x4v3.ncnn.param" "$DIST/realesrgan-x4v3.ncnn.param" \
        --rename in0=input --rename out0=output
    cp "$BUILD/realesrgan_x4v3.ncnn.bin" "$DIST/realesrgan-x4v3.ncnn.bin"
else
    echo "SKIP Real-ESRGAN: missing $WEIGHTS/realesr-general-x4v3.pth"
fi

echo
echo "Models in $DIST:"
ls -lh "$DIST"
echo "Next: tools/pack_models.sh $DIST"
