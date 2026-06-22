#!/usr/bin/env bash
set -euo pipefail
ROOT="${VORTEX_HOME:-/workspace/vortex}"
cd "$ROOT/sim/simx/tensor/open_tensorcore"
make clean
make run-all 2>&1 | tee "$ROOT/otc_sparse_standalone.log"
echo "===== Sparse standalone summary ====="
grep -n "case 4\|case 5\|sparse\|PASS\|FAIL\|max_abs\|threshold" "$ROOT/otc_sparse_standalone.log" || true
