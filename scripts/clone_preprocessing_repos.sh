#!/bin/bash
set -euo pipefail

BASE=/projectnb/cs585/students/mkd/740/WalkVL
EXTERNAL=$BASE/external
mkdir -p "$EXTERNAL"

clone_if_missing() {
    local url="$1"
    local target="$2"
    if [ -d "$target/.git" ]; then
        echo "Already cloned: $target"
        return
    fi
    git clone "$url" "$target"
}

clone_if_missing \
    https://github.com/DepthAnything/Depth-Anything-V2.git \
    "$EXTERNAL/Depth-Anything-V2"

clone_if_missing \
    https://github.com/ansleliu/LightNet.git \
    "$EXTERNAL/LightNet"

echo "Preprocessing repos are under: $EXTERNAL"
