#!/usr/bin/env bash
# Fetches the official ncnn Android Vulkan prebuilt into native/ncnn/<abi>.
# The directory is git-ignored; run this once after cloning.
#
# Override with: NCNN_VERSION=20260526 ABIS="arm64-v8a armeabi-v7a" ./tools/setup_ncnn.sh
# NOTE: keep in sync with PNNX_VERSION in tools/convert/convert_all.sh.
set -euo pipefail

NCNN_VERSION="${NCNN_VERSION:-20261020}"
ABIS="${ABIS:-arm64-v8a}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$PROJECT_ROOT/native/ncnn"
CACHE="$SCRIPT_DIR/.cache"

URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-android-vulkan.zip"
ZIP="$CACHE/ncnn-${NCNN_VERSION}-android-vulkan.zip"

mkdir -p "$CACHE" "$DEST"
if [[ ! -f "$ZIP" ]]; then
    echo "Downloading $URL"
    curl -L --fail --progress-bar -o "$ZIP" "$URL"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q "$ZIP" -d "$TMP"

for ABI in $ABIS; do
    SRC="$TMP/ncnn-${NCNN_VERSION}-android-vulkan/$ABI"
    if [[ ! -d "$SRC" ]]; then
        echo "ERROR: ABI $ABI not found in archive" >&2
        exit 1
    fi
    rm -rf "$DEST/$ABI"
    cp -R "$SRC" "$DEST/$ABI"
    echo "Installed ncnn $NCNN_VERSION -> native/ncnn/$ABI"
done
