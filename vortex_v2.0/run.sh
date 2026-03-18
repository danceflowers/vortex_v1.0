#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
IMAGE_NAME=${IMAGE_NAME:-vortex-v2.0}
CONTAINER_ROOT=/mnt/d/wode_code_trunk/vortex

MOUNT_ROOT=1
if [[ "${1:-}" == "--no-mount" ]]; then
  MOUNT_ROOT=0
  shift
fi

DOCKER_ARGS=(--rm -it)
if [[ $MOUNT_ROOT -eq 1 ]]; then
  DOCKER_ARGS+=(-v "$ROOT_DIR":"$CONTAINER_ROOT" -w "$CONTAINER_ROOT")
fi

if [[ $# -gt 0 ]]; then
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" "$@"
fi

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
