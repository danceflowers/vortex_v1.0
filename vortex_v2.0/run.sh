#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
IMAGE_NAME=${IMAGE_NAME:-vortex-v2.0}

docker run --rm -it \
  -v "$ROOT_DIR":/mnt/d/wode_code_trunk/vortex \
  -w /mnt/d/wode_code_trunk/vortex \
  "$IMAGE_NAME"
