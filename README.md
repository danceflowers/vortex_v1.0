# vortex_v1.0

Minimal standalone package for running these commands:

```bash
make -C hw config CONFIGS="-DEXT_TCU_ENABLE"
make -C runtime/simx clean
make -C runtime/simx CONFIGS="-DEXT_TCU_ENABLE -DNUM_THREADS=32"
```

and

```bash
make -C tests/regression/sgemm_tcu clean && \
make -C tests/regression/sgemm_tcu run-simx \
  CONFIGS="-DEXT_TCU_ENABLE -DNUM_THREADS=32 -DITYPE=fp16 -DOTYPE=fp16" \
  OPTS="-m 128 -n 128 -k 128"
```

## Run with Docker

Build image:

```bash
docker build -t vortex-v1.0 .
```

Open container:

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace vortex-v1.0
```

Inside container, run:

```bash
make -C hw config CONFIGS="-DEXT_TCU_ENABLE"
make -C runtime/simx clean
make -C runtime/simx CONFIGS="-DEXT_TCU_ENABLE -DNUM_THREADS=32"

make -C tests/regression/sgemm_tcu clean && \
make -C tests/regression/sgemm_tcu run-simx \
  CONFIGS="-DEXT_TCU_ENABLE -DNUM_THREADS=32 -DITYPE=fp16 -DOTYPE=fp16" \
  OPTS="-m 128 -n 128 -k 128"
```

## Notes

- `config.mk` defaults `TOOLDIR=/opt/vortex-tools` for Docker.
- The package keeps required prebuilt artifacts `runtime/libvortex.so` and `kernel/libvortex.a` so the above commands run directly.
