#!/usr/bin/env bash
set -euo pipefail

ROOT="${VORTEX_HOME:-/workspace/vortex}"
TEST_DIR="$ROOT/tests/regression/tcgen05_mma_extended"
MF="$TEST_DIR/Makefile"

if [[ ! -d "$ROOT" ]]; then
  echo "[ERR] VORTEX_HOME/root directory not found: $ROOT" >&2
  exit 1
fi
if [[ ! -f "$MF" ]]; then
  echo "[ERR] tcgen05_mma_extended Makefile not found: $MF" >&2
  echo "      Please check whether tests/regression/tcgen05_mma_extended exists." >&2
  exit 1
fi

cd "$TEST_DIR"
cp -n Makefile Makefile.bak_sparse_e2e || true

python3 - <<'PY'
from pathlib import Path
p = Path("Makefile")
s = p.read_text()

# Add runtime rpath for the host executable if the Makefile links with -lvortex.
# This solves: error while loading shared libraries: libvortex.so: cannot open shared object file.
needle = "-L$(VORTEX_HOME)/runtime -lvortex"
repl = "-L$(VORTEX_HOME)/runtime -Wl,-rpath,$(VORTEX_HOME)/runtime -lvortex"
if needle in s and repl not in s:
    s = s.replace(needle, repl)

# If this regression Makefile has no run target, append one.
# It runs the generated host executable in the current test directory and forces the simx driver.
import re
has_run = re.search(r"^run\s*:", s, flags=re.M) is not None
if not has_run:
    s += r'''

# -------------------------------------------------------------------
# Sparse E2E helper targets added by sparse_e2e_tools/patch_tcgen05_regression.sh
# -------------------------------------------------------------------
.PHONY: run run-simx

run: $(PROJECT)
	@echo "[RUN] $(PROJECT) with VORTEX_DRIVER=simx"
	@export VORTEX_HOME=$(VORTEX_HOME); \
	 export VORTEX_DRIVER=simx; \
	 export LD_LIBRARY_PATH=$(VORTEX_HOME)/runtime:$$LD_LIBRARY_PATH; \
	 export PATH=$(VORTEX_HOME)/sim/simx:$$PATH; \
	 ./$(PROJECT)

run-simx: run
'''

p.write_text(s)
print("[OK] Patched tcgen05_mma_extended Makefile")
PY

echo "[OK] Regression Makefile patched: $MF"
