#!/usr/bin/env bash
# Hashes converted models and writes size+sha256 into
# app/src/main/assets/models/models.json (shipped in the APK and used both to
# verify downloads and to configure the native pipeline), then stages all
# artifacts in models-release/ for GitHub Releases.
#
# Usage: tools/pack_models.sh tools/convert/dist
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="${1:-$SCRIPT_DIR/convert/dist}"
MANIFEST="$PROJECT_ROOT/app/src/main/assets/models/models.json"
RELEASE_DIR="$PROJECT_ROOT/models-release"

[[ -d "$SRC" ]] || { echo "model dir not found: $SRC" >&2; exit 1; }

mkdir -p "$RELEASE_DIR"
cp "$SRC"/*.param "$SRC"/*.bin "$RELEASE_DIR"/

python3 - "$RELEASE_DIR" "$MANIFEST" <<'PY'
import hashlib
import json
import os
import sys

release_dir, manifest_path = sys.argv[1], sys.argv[2]
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

for entry in manifest["files"]:
    path = os.path.join(release_dir, entry["path"])
    if not os.path.isfile(path):
        raise SystemExit(f"missing model file: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 18), b""):
            h.update(chunk)
    entry["size"] = os.path.getsize(path)
    entry["sha256"] = h.hexdigest()
    print(f"{entry['id']:>18}  {entry['size'] / 1e6:8.2f} MB  {entry['sha256'][:16]}…")

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

cp "$MANIFEST" "$RELEASE_DIR/models.json"
echo
echo "Staged $RELEASE_DIR"
echo "Publish with: tools/release_models.sh"
