#!/usr/bin/env bash
# Publishes staged models as an immutable GitHub Release. The app downloads
# from the tag URL recorded as base_url in app/src/main/assets/models/models.json
# (set it / override with -PmodelsBaseUrl at build time).
#
# Usage: tools/release_models.sh [tag]   (default: models-1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAG="${1:-models-1}"
DIR="$PROJECT_ROOT/models-release"

command -v gh >/dev/null || { echo "gh CLI required" >&2; exit 1; }
[[ -d "$DIR" ]] || { echo "run tools/pack_models.sh first" >&2; exit 1; }

cd "$PROJECT_ROOT"
gh release create "$TAG" "$DIR"/* \
    --title "Model bundle $TAG" \
    --notes "ncnn models (SCRFD 2.5g, CodeFormer fp16, Real-ESRGAN x4v3). Do not edit in place; publish a new tag instead."
