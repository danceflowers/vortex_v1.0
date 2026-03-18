#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

export VORTEX_HOME="${VORTEX_HOME:-$ROOT_DIR}"
export INSTALLDIR="${INSTALLDIR:-$VORTEX_HOME}"

RUN_TARGET=${RUN_TARGET:-run-simx}
JOBS=${JOBS:-$(nproc)}

ALL_TESTS=(
  sgemm_tcu
  sgemm_tcu_tmem
  tcu_mbarrier_async
  tcu_tmem_chain
  tcu_tmem_shift_fence
  tcu_wmma_overlap
)

usage() {
  echo "Usage: $0 [all|test ...]"
  echo "Tests: ${ALL_TESTS[*]}"
}

select_tests() {
  if [[ $# -eq 0 || "$1" == "all" ]]; then
    printf '%s\n' "${ALL_TESTS[@]}"
    return 0
  fi

  local requested=()
  local test
  for test in "$@"; do
    case "$test" in
      sgemm_tcu|sgemm_tcu_tmem|tcu_mbarrier_async|tcu_tmem_chain|tcu_tmem_shift_fence|tcu_wmma_overlap)
        requested+=("$test")
        ;;
      *)
        echo "Unknown test: $test" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  printf '%s\n' "${requested[@]}"
}

build_runtime() {
  make -C "$ROOT_DIR/hw" config
  make -C "$ROOT_DIR/runtime/simx" clean
  make -C "$ROOT_DIR/runtime/simx" -j"$JOBS"
}

run_test() {
  local test_name="$1"
  local test_dir="$ROOT_DIR/tests/regression/$test_name"

  make -C "$test_dir" clean all

  case "$test_name" in
    sgemm_tcu)
      make -C "$test_dir" "$RUN_TARGET" OPTS="-m 128 -n 128 -k 128"
      ;;
    *)
      make -C "$test_dir" "$RUN_TARGET"
      ;;
  esac
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    return 0
  fi

  mapfile -t selected_tests < <(select_tests "$@")

  build_runtime

  local test_name
  for test_name in "${selected_tests[@]}"; do
    echo "== Running $test_name =="
    run_test "$test_name"
  done
}

main "$@"
