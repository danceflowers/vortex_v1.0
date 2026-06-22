#!/usr/bin/env bash
set -euo pipefail

ROOT="${VORTEX_HOME:-/workspace/vortex}"
TEST_DIR="$ROOT/tests/regression/tcgen05_mma_extended"
LOG="$ROOT/tcgen05_mma_extended_sparse.log"
BUILD_LOG="$ROOT/tcgen05_mma_extended_build.log"
RSP="$ROOT/tcgen05_host_relink.rsp"

export VORTEX_HOME="$ROOT"
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH="$ROOT/sim/simx:$ROOT/runtime:$ROOT/third_party/ramulator:${LD_LIBRARY_PATH:-}"
export PATH="$ROOT/sim/simx:$PATH"

section() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

add_arg() {
  printf '%s\n' "$1" >> "$RSP"
}

append_existing_include() {
  if [ -d "$1" ]; then
    add_arg "-I$1"
  fi
}

append_existing_source() {
  if [ -f "$1" ]; then
    add_arg "$1"
  fi
}

section "[1/8] Check workspace"
echo "ROOT=$ROOT"
echo "TEST_DIR=$TEST_DIR"
cd "$ROOT"

for d in "$ROOT/sim/simx" "$ROOT/runtime" "$ROOT/runtime/simx" "$ROOT/runtime/common" "$TEST_DIR"; do
  if [ ! -d "$d" ]; then
    echo "[ERROR] Directory not found: $d" >&2
    exit 1
  fi
done

section "[2/8] Build simx executable and libsimx.so"
rm -rf "$ROOT/sim/simx/obj"
rm -f "$ROOT/sim/simx/simx" "$ROOT/sim/simx/libsimx.so"
make -C "$ROOT/sim/simx" -j"$(nproc)"
make -C "$ROOT/sim/simx" "$ROOT/sim/simx/libsimx.so" -j"$(nproc)"

if [ ! -x "$ROOT/sim/simx/simx" ]; then
  echo "[ERROR] simx executable was not generated" >&2
  exit 1
fi
if [ ! -f "$ROOT/sim/simx/libsimx.so" ]; then
  echo "[ERROR] libsimx.so was not generated" >&2
  exit 1
fi

section "[3/8] Build SIMX runtime only, skip rtlsim/verilator"
make -C "$ROOT/runtime/simx" clean || true
make -C "$ROOT/runtime/simx" -j"$(nproc)" || true

if [ -f "$ROOT/runtime/libvortex-simx.so" ]; then
  ln -sfn libvortex-simx.so "$ROOT/runtime/libvortex.so"
  ls -lh "$ROOT/runtime"/libvortex*.so || true
else
  echo "[WARN] runtime/libvortex-simx.so was not generated; host will be relinked with runtime source files directly."
fi

section "[4/8] Patch regression Makefile run target if needed"
if [ -x "$ROOT/sparse_tools/patch_tcgen05_regression.sh" ]; then
  "$ROOT/sparse_tools/patch_tcgen05_regression.sh" || true
fi

if ! grep -q '^run:' "$TEST_DIR/Makefile"; then
  cat >> "$TEST_DIR/Makefile" <<'MAKE_EOF'

run:
	LD_LIBRARY_PATH=$(VORTEX_HOME)/runtime:$(VORTEX_HOME)/sim/simx:$$LD_LIBRARY_PATH VORTEX_DRIVER=simx ./tcgen05_mma_extended

run-simx: run
MAKE_EOF
fi

section "[5/8] Build tcgen05_mma_extended kernel; ignore original host link failure"
cd "$TEST_DIR"
make clean || true
set +e
make -k -j"$(nproc)" 2>&1 | tee "$BUILD_LOG"
make_status=${PIPESTATUS[0]}
set -e
if [ "$make_status" -ne 0 ]; then
  echo "[WARN] Normal Makefile build returned non-zero. This is acceptable if kernel.vxbin was generated."
fi
if [ ! -f "$TEST_DIR/kernel.vxbin" ]; then
  echo "[ERROR] kernel.vxbin was not generated. Generated files:" >&2
  find "$TEST_DIR" -maxdepth 2 -type f | sort >&2
  exit 1
fi

section "[6/8] Locate runtime implementation source for vx_* API"
VX_SRC=""
if [ -f "$ROOT/runtime/simx/vortex.cpp" ]; then
  VX_SRC="$ROOT/runtime/simx/vortex.cpp"
fi
if [ -z "$VX_SRC" ]; then
  VX_SRC=$(grep -RIl "vx_dev_open" "$ROOT/runtime" "$ROOT/sim" 2>/dev/null | grep '\.cpp$' | head -1 || true)
fi
if [ -z "$VX_SRC" ] || [ ! -f "$VX_SRC" ]; then
  echo "[ERROR] Could not find a .cpp file implementing vx_dev_open." >&2
  echo "Try: grep -R \"vx_dev_open\" -n $ROOT/runtime $ROOT/sim" >&2
  exit 1
fi
echo "Using VX runtime source: $VX_SRC"

section "[7/8] Force relink host executable using response file"
cd "$TEST_DIR"
rm -f tcgen05_mma_extended
rm -f "$RSP"
: > "$RSP"

# Compiler flags.
add_arg "-std=c++17"
add_arg "-Wall"
add_arg "-Wextra"
add_arg "-pedantic"
add_arg "-Wfatal-errors"
add_arg "-O2"
add_arg "-DNDEBUG"
add_arg "-DEXT_TCU_ENABLE"
add_arg "-DNUM_THREADS=32"

# Basic include directories.
append_existing_include "$ROOT/kernel/include"
append_existing_include "$ROOT/runtime"
append_existing_include "$ROOT/runtime/include"
append_existing_include "$ROOT/runtime/common"
append_existing_include "$ROOT/runtime/simx"
append_existing_include "$ROOT/hw"
append_existing_include "$ROOT/sim/common"
append_existing_include "$ROOT/sim/simx"
append_existing_include "$ROOT/third_party/softfloat/source/include"
append_existing_include "$ROOT/third_party/ramulator/ext/spdlog/include"
append_existing_include "$ROOT/third_party/ramulator/ext/yaml-cpp/include"
append_existing_include "$ROOT/third_party/ramulator/src"

# Add subdirectories under runtime and sim/simx as include directories.
find "$ROOT/runtime" "$ROOT/sim/simx" -type d 2>/dev/null | sort | while IFS= read -r incdir; do
  printf '%s\n' "-I$incdir" >> "$RSP"
done

# Source files for host executable.
append_existing_source "$TEST_DIR/main.cpp"
append_existing_source "$VX_SRC"

# Add runtime/common sources, if any.
find "$ROOT/runtime/common" -maxdepth 1 -type f -name "*.cpp" 2>/dev/null | sort | while IFS= read -r src; do
  printf '%s\n' "$src" >> "$RSP"
done

append_existing_source "$ROOT/sim/common/rvfloats.cpp"
append_existing_source "$ROOT/sim/common/softfloat_ext.cpp"

SOFTFLOAT_A="$ROOT/third_party/softfloat/build/Linux-x86_64-GCC/softfloat.a"
if [ ! -f "$SOFTFLOAT_A" ]; then
  echo "[ERROR] softfloat.a not found: $SOFTFLOAT_A" >&2
  exit 1
fi
add_arg "$SOFTFLOAT_A"

# Libraries and rpath.
add_arg "-L$ROOT/sim/simx"
add_arg "-Wl,--no-as-needed"
add_arg "-lsimx"
add_arg "-Wl,--as-needed"

if [ -d "$ROOT/third_party/ramulator" ]; then
  add_arg "-L$ROOT/third_party/ramulator"
  add_arg "-lramulator"
  add_arg "-Wl,-rpath,$ROOT/third_party/ramulator"
fi

add_arg "-Wl,-rpath,$ROOT/sim/simx"
add_arg "-pthread"
add_arg "-ldl"
add_arg "-o"
add_arg "$TEST_DIR/tcgen05_mma_extended"

echo "Host relink response file: $RSP"
echo "First 120 lines of response file:"
sed -n '1,120p' "$RSP"

g++ @"$RSP"

if [ ! -x "$TEST_DIR/tcgen05_mma_extended" ]; then
  echo "[ERROR] host executable was not generated." >&2
  exit 1
fi
ls -lh "$TEST_DIR/tcgen05_mma_extended"

section "[8/8] Run sparse regression with SIMX"
cd "$TEST_DIR"
set +e
LD_LIBRARY_PATH="$ROOT/sim/simx:$ROOT/runtime:$ROOT/third_party/ramulator:${LD_LIBRARY_PATH:-}" \
VORTEX_DRIVER=simx \
"$TEST_DIR/tcgen05_mma_extended" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "============================================================"
echo "Sparse regression result"
echo "============================================================"
echo "Program exit status: $status"
echo "Log: $LOG"
grep -n "sparse_2_4\|sparse_1_4\|2:4\|1:4\|PASS\|FAIL\|max_abs\|threshold\|aggregate\|ALL" "$LOG" || true

if [ "$status" -ne 0 ]; then
  echo "[FAIL] tcgen05_mma_extended returned non-zero status" >&2
  exit "$status"
fi
if grep -q "FAIL" "$LOG"; then
  echo "[FAIL] Found FAIL in regression log" >&2
  exit 1
fi

echo "[PASS] tcgen05_mma_extended finished without FAIL"
echo "Please confirm that both 2:4 and 1:4 sparse cases are shown as PASS in the log."
