# vortex_v2.0

Docker packaging for the current Vortex CModel tree and the latest TCU-related regression tests. The package keeps the in-container source path fixed at:

```bash
/mnt/d/wode_code_trunk/vortex
```

That lets you run the existing `make -C ...` and `LD_LIBRARY_PATH=... VORTEX_DRIVER=simx ...` commands without changing paths.

## Build the image

Run this from the repository root:

```bash
docker build -t vortex-v2.0 -f vortex_v2.0/Dockerfile .
```

The image build can bootstrap missing `third_party/softfloat`, `third_party/ramulator`,
and `third_party/ramulator/ext/*` sources, so a clean checkout can still build the image.

Before building on another server, sync the repository and submodules:

```bash
git pull
git submodule update --init --recursive
```

## Open a container

If you want the container to use the checkout you pulled from GitHub, bind-mount the repository into the same in-container path:

```bash
./vortex_v2.0/run.sh
```

If you bind-mount the host checkout, it hides the snapshot baked into the image.
Make sure the host tree itself has the required submodules populated, for example:

```bash
git submodule update --init --recursive
```

If you only want the image snapshot that was baked in during `docker build`, drop the `-v` mount:

```bash
./vortex_v2.0/run.sh --no-mount
```

You can also run a command directly without opening an interactive shell:

```bash
./vortex_v2.0/run.sh ./vortex_v2.0/run_tcu_tests.sh
```

## Commands to run inside the container

Prepare the generated hardware headers and rebuild the CModel:

```bash
make -C /mnt/d/wode_code_trunk/vortex/hw config
make -C /mnt/d/wode_code_trunk/vortex/runtime/simx clean
make -C /mnt/d/wode_code_trunk/vortex/runtime/simx -j"$(nproc)"
```

Run all current TCU-related regressions in one shot:

```bash
./vortex_v2.0/run_tcu_tests.sh
```

Run a subset:

```bash
./vortex_v2.0/run_tcu_tests.sh sgemm_tcu sgemm_tcu_tmem
```

## Individual TCU Tests

Baseline SGEMM TCU:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/sgemm_tcu clean all && \
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/sgemm_tcu run-simx \
  OPTS="-m 128 -n 128 -k 128"
```

TMEM/TMA SGEMM:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/sgemm_tcu_tmem clean all run-simx
```

Single-tile TMEM chain:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/tcu_tmem_chain clean all run-simx
```

Async mbarrier + TMEM:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/tcu_mbarrier_async clean all run-simx
```

TMEM shift fence:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/tcu_tmem_shift_fence clean all run-simx
```

WMMA overlap:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/tcu_wmma_overlap clean all run-simx
```

## Notes

- The image installs the same dependency/toolchain flow used by `vortex_v1.0`: `ci/install_dependencies.sh` and `ci/toolchain_install.sh`.
- The current TCU regression set is `sgemm_tcu`, `sgemm_tcu_tmem`, `tcu_mbarrier_async`, `tcu_tmem_chain`, `tcu_tmem_shift_fence`, and `tcu_wmma_overlap`.
- The default flow assumes the current repository defaults: `NUM_THREADS=32` and `EXT_TCU_ENABLE=1`.
