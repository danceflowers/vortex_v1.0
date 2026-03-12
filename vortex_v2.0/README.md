# vortex_v2.0

Docker packaging for the current Vortex CModel tree. The package follows the same pattern as `vortex_v1.0`, but it keeps the in-container source path fixed at:

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

## Open a container

If you want the container to use the checkout you pulled from GitHub, bind-mount the repository into the same in-container path:

```bash
docker run --rm -it \
  -v "$PWD":/mnt/d/wode_code_trunk/vortex \
  -w /mnt/d/wode_code_trunk/vortex \
  vortex-v2.0
```

If you bind-mount the host checkout, it hides the snapshot baked into the image.
Make sure the host tree itself has the required submodules populated, for example:

```bash
git submodule update --init --recursive
```

If you only want the image snapshot that was baked in during `docker build`, drop the `-v` mount:

```bash
docker run --rm -it vortex-v2.0
```

## Commands to run inside the container

Rebuild the CModel:

```bash
make -C /mnt/d/wode_code_trunk/vortex/runtime/simx clean
make -C /mnt/d/wode_code_trunk/vortex/runtime/simx -j4
```

Build and run the new `128x128x128` TMEM/TMA regression:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/sgemm_tcu_tmem clean all
cd /mnt/d/wode_code_trunk/vortex/tests/regression/sgemm_tcu_tmem
LD_LIBRARY_PATH=/mnt/d/wode_code_trunk/vortex/runtime \
VORTEX_DRIVER=simx \
./sgemm_tcu_tmem
```

Build and run the single-tile TMEM/TMA chain regression:

```bash
make -C /mnt/d/wode_code_trunk/vortex/tests/regression/tcu_tmem_chain clean all
cd /mnt/d/wode_code_trunk/vortex/tests/regression/tcu_tmem_chain
LD_LIBRARY_PATH=/mnt/d/wode_code_trunk/vortex/runtime \
VORTEX_DRIVER=simx \
./tcu_tmem_chain
```

## Notes

- The image installs the same dependency/toolchain flow used by `vortex_v1.0`: `ci/install_dependencies.sh` and `ci/toolchain_install.sh`.
- `TMEM/TMA` regressions assume the current repository defaults: `NUM_THREADS=32` and `EXT_TCU_ENABLE=1`.
